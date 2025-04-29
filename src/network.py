# --------------------------------------------------------------------------- #
#                                src/network.py                               #
# --------------------------------------------------------------------------- #
from typing import Dict, Any

import copy
import torch
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from termcolor import colored

from src.metric import MyAccuracy          # 필요하다면 MyF1Score 도 import
import src.config as cfg
from src.util import show_setting


# ------------------------ 1.  Squeeze & Excitation ------------------------ #
class _SE(nn.Module):
    """Lightweight Squeeze-and-Excitation block (optional)."""

    def __init__(self, channels: int, rd_ratio: float = 1 / 16):
        super().__init__()
        rd_channels = max(8, int(channels * rd_ratio))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, rd_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rd_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg(x).flatten(1)
        w = self.fc(w).view(x.size(0), -1, 1, 1)
        return x * w


# ------------------------ 2.  Custom Alex-Net Variant --------------------- #
class MyNetwork(AlexNet):
    """
    Modernised AlexNet.

    • Conv 커널을 7→3 로 축소 & stride/padding 업데이트  
    • LRN 제거, BatchNorm + Dropout 도입  
    • SE 블록(옵션)  
    • AdaptiveAvgPool((1,1)) 로 FC 파라미터 대폭 감소
    """

    USE_SE = True  # 성능 ↔ 속도 균형에 따라 조절

    def __init__(self, num_classes: int = cfg.NUM_CLASSES, dropout: float = 0.5):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # -------- Feature Extractor (새로 작성) -------------------------- #
        self.features = nn.Sequential(
            # Stage 1 -------------------------------------------------- #
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.2),

            # Stage 2 -------------------------------------------------- #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.2),

            # Stage 3 -------------------------------------------------- #
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 선택적 Squeeze-and-Excitation
        if self.USE_SE:
            self.features.add_module("se", _SE(512))

        # Adaptive pool (6×6 → 1×1) 로 변경, 덕분에 FC 입력 = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # -------- Classifier (대폭 경량화) ------------------------------ #
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    # 필요 시 forward 재정의 (여기선 base 와 동일)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------------- #
#                       3.  LightningModule Wrapper                           #
# --------------------------------------------------------------------------- #
class SimpleClassifier(LightningModule):
    """
    • model_name == 'MyNetwork' 이면 위 정의한 네트워크 사용  
    • 그 외에는 torchvision.get_model()
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = cfg.NUM_CLASSES,
        optimizer_params: Dict[str, Any] | None = None,
        scheduler_params: Dict[str, Any] | None = None,
    ):
        super().__init__()

        # ------------------------ network --------------------------- #
        if model_name == "MyNetwork":
            self.model = MyNetwork(num_classes=num_classes)
        else:
            models_list = models.list_models()
            assert (
                model_name in models_list
            ), f"Unknown model name: {model_name}. Choose from {', '.join(models_list)}"
            self.model = models.get_model(model_name, num_classes=num_classes)

        # ------------------------ loss / metrics -------------------- #
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = MyAccuracy()
        # self.f1 = MyF1Score(num_classes=num_classes)  # 필요 시 활성화

        # ------------------------ hparams save ---------------------- #
        self.save_hyperparameters(
            dict(
                model_name=model_name,
                num_classes=num_classes,
                optimizer_params=optimizer_params or {},
                scheduler_params=scheduler_params or {},
            )
        )

    # -------------------  Lightning hooks  ------------------------ #
    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop("type")
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        sched_params = copy.deepcopy(self.hparams.scheduler_params)
        sched_type = sched_params.pop("type")
        scheduler = getattr(torch.optim.lr_scheduler, sched_type)(optimizer, **sched_params)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # -------------------  forward / steps  ------------------------ #
    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch)
        acc = self.accuracy(logits, y)
        self.log_dict({"loss/train": loss, "accuracy/train": acc},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch)
        acc = self.accuracy(logits, y)
        self.log_dict({"loss/val": loss, "accuracy/val": acc},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, logits, frequency=cfg.WANDB_IMG_LOG_FREQ)

    # -------------------  misc util  ------------------------------ #
    def _wandb_log_image(self, batch, batch_idx, preds, frequency: int = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", "blue", attrs=("bold",)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f"pred/val/batch{batch_idx:05d}_sample_0",
                images=[x[0].to("cpu")],
                caption=[f"GT: {y[0].item()}, Pred: {preds[0].item()}"],
            )
