# src/metric.py
from typing import Optional, Union, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric


# --------------------------------------------------------------------------- #
#                           1.  F-1  SCORE  (ONE-VS-REST)                     #
# --------------------------------------------------------------------------- #
class MyF1Score(Metric):


    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-8,
        dist_sync_on_step: bool = False,
    ) -> None:
        """
        Args
        ----
        num_classes : 전체 클래스 수.
        eps         : 0 으로 나누는 상황을 방지하기 위한 작은 값.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.eps = eps

        # 상태 변수들 ― 분산 환경에서도 자동 합산(dist-reduce) 되도록 설정
        self.add_state(
            "tp",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    # --------------------------------------------------------------------- #
    #                              UPDATE                                   #
    # --------------------------------------------------------------------- #
    def update(self, preds: Tensor, target: Tensor) -> None:  # noqa: D401

        if preds.ndim != 2:
            raise ValueError(
                f"`preds` must have shape (batch, num_classes); got {preds.shape}"
            )
        if preds.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {preds.shape[1]}"
            )
        if target.ndim != 1:
            raise ValueError(
                f"`target` must be 1-D tensor of class indices; got {target.shape}"
            )

        preds_idx = torch.argmax(preds, dim=1)

        # 각 클래스별 TP/FP/FN 계산 (벡터화)
        for c in range(self.num_classes):
            pred_c = preds_idx == c
            true_c = target == c

            self.tp[c] += torch.sum(pred_c & true_c)
            self.fp[c] += torch.sum(pred_c & ~true_c)
            self.fn[c] += torch.sum(~pred_c & true_c)

    # --------------------------------------------------------------------- #
    #                              COMPUTE                                  #
    # --------------------------------------------------------------------- #
    def compute(self) -> Tensor:
        """
        Return
        ------
        Tensor (num_classes,) : 각 클래스의 F-1 점수.
        """
        precision = self.tp.float() / (self.tp + self.fp + self.eps)
        recall = self.tp.float() / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        return f1


# --------------------------------------------------------------------------- #
#                                  2.  ACCURACY                               #
# --------------------------------------------------------------------------- #
class MyAccuracy(Metric):
    """
    단순 분류 정확도(정답 / 전체) 구현.
    """

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total",   default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    # --------------------------------------------------------------------- #
    #                                UPDATE                                 #
    # --------------------------------------------------------------------- #
    def update(self, preds: Tensor, target: Tensor) -> None:

        if preds.ndim != 2:
            raise ValueError(f"`preds` must be 2-D (B, C); got {preds.shape}")
        if target.ndim != 1:
            raise ValueError(f"`target` must be 1-D (B,); got {target.shape}")

        preds_idx = torch.argmax(preds, dim=1)

        if preds_idx.shape != target.shape:
            raise ValueError(
                f"Shape mismatch after argmax: {preds_idx.shape} vs {target.shape}"
            )

        self.correct += torch.sum(preds_idx == target)
        self.total   += target.numel()

    # --------------------------------------------------------------------- #
    #                               COMPUTE                                 #
    # --------------------------------------------------------------------- #
    def compute(self) -> Tensor:
        """
        Returns
        -------
        Tensor : 정확도 스칼라 (float32). 데이터가 없다면 NaN.
        """
        if self.total == 0:
            # torch.nan_to_num 등으로 후처리 가능
            return torch.tensor(float("nan"), device=self.total.device)
        return self.correct.float() / self.total.float()
