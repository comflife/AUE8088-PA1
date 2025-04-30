"""
[AUE8088] PA1: Image Classification – *Multi-run & Trade-off Plotting edition*

$ python train.py                     # cfg.MODEL_NAME 1회 실행
$ python train.py efficientnet_b0 ... # 인수로 준 모델들 순차 실행
"""

# --------------------------------------------------------------------------- #
#                               0.  IMPORTS                                   #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
import matplotlib.pyplot as plt
import wandb

# -------------------------------- my modules ------------------------------ #
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

torch.set_float32_matmul_precision("medium")
wandb.login(key="")

# --------------------------------------------------------------------------- #
#                   1.  SINGLE-RUN HELPER (train + validate)                  #
# --------------------------------------------------------------------------- #
def run_single(model_name: str) -> Dict[str, Any]:
    """train + validate one model, return dict with stats."""
    seed_everything(42, workers=True)

    # 1) model / data ------------------------------------------------------ #
    model = SimpleClassifier(
        model_name=model_name,
        num_classes=cfg.NUM_CLASSES,
        optimizer_params=cfg.OPTIMIZER_PARAMS,
        scheduler_params=cfg.SCHEDULER_PARAMS,
    )
    datamodule = TinyImageNetDatasetModule(batch_size=cfg.BATCH_SIZE)

    # 2) logger ------------------------------------------------------------ #
    wandb_logger = WandbLogger(
        project=cfg.WANDB_PROJECT,
        save_dir=cfg.WANDB_SAVE_DIR,
        entity=cfg.WANDB_ENTITY,
        name=f"{cfg.WANDB_NAME}-{model_name}",
    )

    # 3) trainer ----------------------------------------------------------- #
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

    # 4) fit + validate ---------------------------------------------------- #
    trainer.fit(model, datamodule=datamodule)
    val_metrics = trainer.validate(ckpt_path="best", datamodule=datamodule)[0]

    # 5) stats ------------------------------------------------------------- #
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # M
    stats = dict(
        model=model_name,
        params_m=n_params,
        top1_acc=float(val_metrics["accuracy/val"]),
    )
    return stats


# --------------------------------------------------------------------------- #
#                        2.  MULTI-RUN + PLOTTING                             #
# --------------------------------------------------------------------------- #
def multi_run(model_list: List[str]) -> None:
    """loop over model_list, save csv & plot."""
    results: List[Dict[str, Any]] = []
    for m in model_list:
        print(f"\n=== Training {m} ===")
        stats = run_single(m)
        print(f" → finished {m}:  {stats}")
        results.append(stats)

    # 1) write CSV --------------------------------------------------------- #
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "tradeoff.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "params_m", "top1_acc"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved summary to {csv_path.absolute()}")

    # 2) plot -------------------------------------------------------------- #
    xs = [r["params_m"] for r in results]
    ys = [r["top1_acc"] * 100 for r in results]
    labels = [r["model"] for r in results]

    plt.figure()
    plt.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, lab)
    plt.xlabel("# Params (Millions)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Size-Accuracy Trade-off (TinyImageNet-200)")
    plt.grid(True)
    plot_path = out_dir / "tradeoff.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path.absolute()}\n")


# --------------------------------------------------------------------------- #
#                               3.  MAIN                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        help="model names (e.g. efficientnet_b0 efficientnet_b1 …). "
        "비우면 cfg.MODEL_NAME 1개만 실행",
    )
    args = parser.parse_args()

    models_to_run = args.models or [cfg.MODEL_NAME]
    multi_run(models_to_run)
