# """
# $ python train.py                     # cfg.MODEL_NAME 1회 실행
# $ python train.py efficientnet_b0 ... # 인수로 준 모델들 순차 실행
# """

# # --------------------------------------------------------------------------- #
# #                               0.  IMPORTS                                   #
# # --------------------------------------------------------------------------- #
# from __future__ import annotations
# import argparse
# import csv
# from pathlib import Path
# from typing import List, Dict, Any

# import torch
# from lightning import Trainer, seed_everything
# from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
# from lightning.pytorch.loggers.wandb import WandbLogger
# import matplotlib.pyplot as plt
# import wandb

# # -------------------------------- my modules ------------------------------ #
# from src.dataset import TinyImageNetDatasetModule
# from src.network import SimpleClassifier
# import src.config as cfg

# torch.set_float32_matmul_precision("medium")
# wandb.login(key="")

# # --------------------------------------------------------------------------- #
# #                   1.  SINGLE-RUN HELPER (train + validate)                  #
# # --------------------------------------------------------------------------- #
# def run_single(model_name: str) -> Dict[str, Any]:
#     """train + validate one model, return dict with stats."""
#     seed_everything(42, workers=True)

#     # 1) model / data ------------------------------------------------------ #
#     model = SimpleClassifier(
#         model_name=model_name,
#         num_classes=cfg.NUM_CLASSES,
#         optimizer_params=cfg.OPTIMIZER_PARAMS,
#         scheduler_params=cfg.SCHEDULER_PARAMS,
#     )
#     datamodule = TinyImageNetDatasetModule(batch_size=cfg.BATCH_SIZE)

#     # 2) logger ------------------------------------------------------------ #
#     wandb_logger = WandbLogger(
#         project=cfg.WANDB_PROJECT,
#         save_dir=cfg.WANDB_SAVE_DIR,
#         entity=cfg.WANDB_ENTITY,
#         name=f"{cfg.WANDB_NAME}-{model_name}",
#     )

#     # 3) trainer ----------------------------------------------------------- #
#     trainer = Trainer(
#         accelerator=cfg.ACCELERATOR,
#         devices=cfg.DEVICES,
#         precision=cfg.PRECISION_STR,
#         max_epochs=cfg.NUM_EPOCHS,
#         check_val_every_n_epoch=cfg.VAL_EVERY_N_EPOCH,
#         logger=wandb_logger,
#         callbacks=[
#             LearningRateMonitor(logging_interval="epoch"),
#             ModelCheckpoint(save_top_k=1, monitor="accuracy/val", mode="max"),
#         ],
#         enable_progress_bar=True,
#     )

#     # 4) fit + validate ---------------------------------------------------- #
#     trainer.fit(model, datamodule=datamodule)
#     val_metrics = trainer.validate(ckpt_path="best", datamodule=datamodule)[0]

#     # 5) stats ------------------------------------------------------------- #
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # M
#     stats = dict(
#         model=model_name,
#         params_m=n_params,
#         top1_acc=float(val_metrics["accuracy/val"]),
#     )
#     return stats


# # --------------------------------------------------------------------------- #
# #                        2.  MULTI-RUN + PLOTTING                             #
# # --------------------------------------------------------------------------- #
# def multi_run(model_list: List[str]) -> None:
#     """loop over model_list, save csv & plot."""
#     results: List[Dict[str, Any]] = []
#     for m in model_list:
#         print(f"\n=== Training {m} ===")
#         stats = run_single(m)
#         print(f" → finished {m}:  {stats}")
#         results.append(stats)

#     # 1) write CSV --------------------------------------------------------- #
#     out_dir = Path("results")
#     out_dir.mkdir(exist_ok=True)
#     csv_path = out_dir / "tradeoff.csv"
#     with csv_path.open("w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["model", "params_m", "top1_acc"])
#         writer.writeheader()
#         writer.writerows(results)
#     print(f"\nSaved summary to {csv_path.absolute()}")

#     # 2) plot -------------------------------------------------------------- #
#     xs = [r["params_m"] for r in results]
#     ys = [r["top1_acc"] * 100 for r in results]
#     labels = [r["model"] for r in results]

#     plt.figure()
#     plt.scatter(xs, ys)
#     for x, y, lab in zip(xs, ys, labels):
#         plt.text(x, y, lab)
#     plt.xlabel("# Params (Millions)")
#     plt.ylabel("Top-1 Accuracy (%)")
#     plt.title("Size-Accuracy Trade-off (TinyImageNet-200)")
#     plt.grid(True)
#     plot_path = out_dir / "tradeoff.png"
#     plt.savefig(plot_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"Saved plot to {plot_path.absolute()}\n")


# # --------------------------------------------------------------------------- #
# #                               3.  MAIN                                      #
# # --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "models",
#         nargs="*",
#         help="model names (e.g. efficientnet_b0 efficientnet_b1 …). "
#         "비우면 cfg.MODEL_NAME 1개만 실행",
#     )
#     args = parser.parse_args()

#     models_to_run = args.models or [cfg.MODEL_NAME]
#     multi_run(models_to_run)

"""
[AUE8088] PA1: Image Classification – Multi-run & RandAug Sweep edition
-----------------------------------------------------------------------
6개의 EfficientNet 계열 모델 × 4개 RandAug 파라미터 조합
→ 총 24회 학습 & 자동 W&B 그룹/이름 분리
"""

# -------------------------------------------------------------------- #
# 0. IMPORTS                                                           #
# -------------------------------------------------------------------- #
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch, matplotlib.pyplot as plt, wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from src.dataset import TinyImageNetDatasetModule        
from src.network import SimpleClassifier
import src.config as cfg

torch.set_float32_matmul_precision("medium")
wandb.login(key="")

# -------------------------------------------------------------------- #
# 1. SINGLE-RUN (model, aug_n, aug_m)                                   #
# -------------------------------------------------------------------- #
def run_single(model_name: str, aug_nm: Tuple[int, int]) -> Dict[str, Any]:
    """train+validate one (model, RandAug) combo and return stats dict"""
    aug_n, aug_m = aug_nm
    seed_everything(42, workers=True)

    # 1) model & data -------------------------------------------------- #
    model = SimpleClassifier(
        model_name=model_name,
        num_classes=cfg.NUM_CLASSES,
        optimizer_params=cfg.OPTIMIZER_PARAMS,
        scheduler_params=cfg.SCHEDULER_PARAMS,
    )
    datamodule = TinyImageNetDatasetModule(
        batch_size=cfg.BATCH_SIZE,
        aug_n=aug_n,
        aug_m=aug_m,
    )

    # 2) logger (W&B) -------------------------------------------------- #
    wandb_logger = WandbLogger(
        project=cfg.WANDB_PROJECT,
        save_dir=cfg.WANDB_SAVE_DIR,
        entity=cfg.WANDB_ENTITY,
        group=model_name,                         # 모델 단위로 그룹
        name=f"{model_name}-N{aug_n}_M{aug_m}",   # 모델+증강 조합 Run 이름
        reinit=True,  
    )

    # 3) trainer ------------------------------------------------------- #
    trainer = Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICES,
        precision=cfg.PRECISION_STR,
        max_epochs=cfg.NUM_EPOCHS,
        check_val_every_n_epoch=cfg.VAL_EVERY_N_EPOCH,
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(save_top_k=1, monitor="accuracy/val", mode="max"),
        ],
        enable_progress_bar=True,
    )

    # 4) fit + validate ------------------------------------------------ #
    trainer.fit(model, datamodule=datamodule)
    val_metrics = trainer.validate(ckpt_path="best", datamodule=datamodule)[0]
    wandb.finish()  

    # 5) stats dict ---------------------------------------------------- #
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return dict(
        model=model_name,
        aug_n=aug_n,
        aug_m=aug_m,
        params_m=n_params,
        top1_acc=float(val_metrics["accuracy/val"]),
    )


# -------------------------------------------------------------------- #
# 2. MULTI-RUN                                                         #
# -------------------------------------------------------------------- #
def multi_run(model_list: List[str]) -> None:
    aug_grid = [(5, 15), (8, 15), (5, 20), (8, 20)]   # N,M 조합
    results: List[Dict[str, Any]] = []

    for m in model_list:
        for am in aug_grid:
            print(f"\n=== Training {m} | RandAug N={am[0]}, M={am[1]} ===")
            stats = run_single(m, am)
            print(f" → finished: {stats}")
            results.append(stats)

    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "tradeoff.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "aug_n", "aug_m", "params_m", "top1_acc"],
        )
        writer.writeheader(); writer.writerows(results)
    print(f"\nSaved summary to {csv_path.absolute()}")

    xs = [r["params_m"] for r in results]
    ys = [r["top1_acc"] * 100 for r in results]
    labs = [f"{r['model']}\nN{r['aug_n']} M{r['aug_m']}" for r in results]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labs):
        plt.text(x, y, lab, fontsize=7)
    plt.xlabel("# Params (M)"); plt.ylabel("Top-1 Acc (%)")
    plt.title("Size vs Accuracy (RandAug sweep)")
    plt.grid(True)
    plot_path = out_dir / "tradeoff.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved plot to {plot_path.absolute()}\n")


# -------------------------------------------------------------------- #
# 3. MAIN                                                              #
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    eff_models = [
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "efficientnet_b3", "efficientnet_v2_s", "efficientnet_v2_m",
    ]

    # optional: CLI로 특정 모델만 골라 돌리고 싶을 때
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="subset models to run")
    args = parser.parse_args()
    models_to_run = args.models or eff_models

    multi_run(models_to_run)
