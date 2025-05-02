# --------------------------------------------------------------------------- #
#                                src/config.py                                #
# --------------------------------------------------------------------------- #
import os
from datetime import datetime

# --------------------------------------------------------------------------- #
# 1) DATA                                                                     #
# --------------------------------------------------------------------------- #
DATASET_ROOT_PATH   = "datasets/"
NUM_WORKERS         = 8

# Augmentation (Train/Val transforms 정의 파일에서 읽어갈 변수들)
IMAGE_ROTATION      = 30             
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

RAND_AUG_N          = 2               # 연산 개수
RAND_AUG_M          = 9               # 강도

# --------------------------------------------------------------------------- #
# 2) TRAINING HYPER-PARAMETERS                                                #
# --------------------------------------------------------------------------- #
NUM_CLASSES         = 200
BATCH_SIZE          = 384             #
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 60          

# Optimizer & Scheduler ---------------------------------------------------- #
OPTIMIZER_PARAMS = {
    "type": "AdamW",
    "lr": 3e-4,                       # 디폴트 β=(0.9, 0.999) 는 모듈 측에서 설정
    "weight_decay": 5e-2,
}

SCHEDULER_PARAMS = {
    "type": "CosineAnnealingLR",
    "T_max": NUM_EPOCHS,
    "eta_min": 1e-6,
}

# --------------------------------------------------------------------------- #
# 3) NETWORK / MODEL                                                          #
# --------------------------------------------------------------------------- #
MODEL_NAME = "resnet34"              


LABEL_SMOOTHING    = 0.1

# --------------------------------------------------------------------------- #
# 4) COMPUTE SETTINGS                                                         #
# --------------------------------------------------------------------------- #
ACCELERATOR   = "gpu"
DEVICES       = [0]                   # 여러 장이면 [0,1,2,…]
PRECISION_STR = "16-mixed"            # fp16 mixed precision

# --------------------------------------------------------------------------- #
# 5) LOGGING                                                                  #
# --------------------------------------------------------------------------- #
WANDB_PROJECT      = "aue8088-pa1"
WANDB_ENTITY       = os.environ.get("WANDB_ENTITY")
WANDB_SAVE_DIR     = "wandb/"
WANDB_IMG_LOG_FREQ = 50

_run_time          = datetime.now().strftime("%m%d_%H%M")
WANDB_NAME         = (
    f"{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS['type']}"
    f"-CosLR{OPTIMIZER_PARAMS['lr']:.0e}-{_run_time}"
)
