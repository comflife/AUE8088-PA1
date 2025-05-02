from termcolor import colored
from tqdm import tqdm
import os
import tarfile
import wget

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandAugment   # ★ NEW


import src.config as cfg


class TinyImageNetDatasetModule(LightningDataModule):
    __DATASET_NAME__ = 'tiny-imagenet-200'

    # def __init__(self, batch_size: int = cfg.BATCH_SIZE):
    #     super().__init__()
    #     self.batch_size = batch_size

    def __init__(
        self,
        batch_size: int = cfg.BATCH_SIZE,
        aug_n: int = cfg.RAND_AUG_N,
        aug_m: int = cfg.RAND_AUG_M,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.aug_n = aug_n        # 저장
        self.aug_m = aug_m
        

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        if not os.path.exists(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__)):
            # download data
            print(colored("\nDownloading dataset...", color='green', attrs=('bold',)))
            filename = self.__DATASET_NAME__ + '.tar'
            wget.download(f'https://hyu-aue8088.s3.ap-northeast-2.amazonaws.com/{filename}')

            # extract data
            print(colored("\nExtract dataset...", color='green', attrs=('bold',)))
            with tarfile.open(name=filename) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(path=cfg.DATASET_ROOT_PATH, member=member)
            os.remove(filename)

    def train_dataloader(self):
        tf_train = transforms.Compose([
            RandAugment(num_ops=self.aug_n, magnitude=self.aug_m),
            transforms.RandomRotation(cfg.IMAGE_ROTATION),
            transforms.RandomHorizontalFlip(cfg.IMAGE_FLIP_PROB),
            transforms.RandomCrop(cfg.IMAGE_NUM_CROPS, padding=cfg.IMAGE_PAD_CROPS),
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'train'), tf_train)
        msg = f"[Train]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'val'), tf_val)
        msg = f"[Val]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'test'), tf_test)
        msg = f"[Test]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )
