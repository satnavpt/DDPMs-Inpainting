import os

import torch
from torchvision import transforms as tvt, datasets as tvds
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from PIL import Image
import cv2

from core.utils import ConditionType
from core.mask import bbox2mask, get_irregular_mask, random_bbox, random_cropping_bbox

ROOT = os.path.expanduser("./datasets")


def inverse_scaler(x):
    return x


class CIFAR10(tvds.CIFAR10):
    name = "cifar10"
    resolution = (32, 32)
    channels = 3
    all_size = None
    train_size = 50000
    test_size = 10000
    flip = tvt.RandomHorizontalFlip()
    _transform = tvt.PILToTensor()
    condition = ConditionType.none

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
        uniform_dequantisation: bool = False,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if uniform_dequantisation:
            self.tensorise = tvt.ToTensor()
            self.transform = tvt.Lambda(lambda x: (x * 2.0) - 1.0)
        else:
            self.transform = tvt.Compose(
                [tvt.ToTensor(), tvt.Lambda(lambda x: (x * 2.0) - 1.0)]
            )

        self.uniform_dequantisation = uniform_dequantisation

    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img, mode="RGB")
        img = self.flip(img)
        if self.uniform_dequantisation:
            img = self.tensorise(img)
            img = (torch.rand(img.shape, dtype=torch.float32) + img * 255.0) / 256.0

        ret = {}

        ret["gt"] = self.transform(img)

        return ret


class CIFAR10Colourisation(CIFAR10):
    condition = ConditionType.colourisation

    def __getitem__(self, index: int):
        img, _ = self.data[index], self.targets[index]

        img = Image.fromarray(img, mode="RGB")
        img = self.flip(img)
        grey = Image.fromarray(
            cv2.cvtColor(np.array(img.convert("L")), cv2.COLOR_GRAY2RGB)
        )

        if self.uniform_dequantisation:
            img, grey = self.tensorise(img), self.tensorise(grey)
            img = (torch.rand(img.shape, dtype=torch.float32) + img * 255.0) / 256.0
            grey = (torch.rand(grey.shape, dtype=torch.float32) + grey * 255.0) / 256.0

        ret = {}
        ret["gt"] = self.transform(img)
        ret["cond"] = self.transform(grey)

        return ret


class CelebAHQ(tvds.VisionDataset):
    name = "celebAHQ"
    base_folder = "celebAHQ"
    resolution = (256, 256)
    channels = 3
    flip = tvt.RandomHorizontalFlip()
    _transform = tvt.PILToTensor()
    all_size = 30000
    train_size = 30000
    test_size = 5000
    condition = ConditionType.none

    def __init__(
        self, root, transform=None, split="train", uniform_dequantisation=False
    ):
        super().__init__(root, transform=transform or self._transform)
        self.split = split
        if self.split == "train":
            splitList = range(1, self.train_size + 1)
        elif self.split == "all":
            splitList = range(1, self.all_size + 1)
        elif self.split == "test":
            splitList = range(self.all_size - self.test_size, self.all_size + 1)
        self.filename = sorted(
            [
                fname
                for fname in os.listdir(os.path.join(root, self.base_folder))
                if fname.endswith(".jpg") and int(fname.split(".")[0]) in splitList
            ],
            key=lambda name: int(name[:-4].zfill(5)),
        )
        np.random.RandomState(123).shuffle(self.filename)

        if uniform_dequantisation:
            self.tensorise = tvt.ToTensor()
            self.transform = tvt.Lambda(lambda x: (x * 2.0) - 1.0)
        else:
            self.transform = tvt.Compose(
                [tvt.ToTensor(), tvt.Lambda(lambda x: (x * 2.0) - 1.0)]
            )

        self.uniform_dequantisation = uniform_dequantisation

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(self.root, self.base_folder, self.filename[index])
        )
        img = self.flip(img)
        if self.uniform_dequantisation:
            img = self.tensorise(img)
            img = (torch.rand(img.shape, dtype=torch.float32) + img * 255.0) / 256.0

        ret = {}
        ret["gt"] = self.transform(img)

        return ret

    def __len__(self):
        return len(self.filename)


class CelebAHQInpainting(CelebAHQ):
    name = "celebAHQInpainting"
    condition = ConditionType.inpainting
    mask_config = {"mask_mode": "boxAndFreeForm"}
    mask_mode = mask_config["mask_mode"]
    maxBboxMaskSize = (100, 100)
    maxBboxMaskDelta = (75, 75)
    minMargin = 25
    freeFormRatio = (0.2, 0.4)

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(self.root, self.base_folder, self.filename[index])
        )

        img = self.flip(img)

        if self.uniform_dequantisation:
            img = self.tensorise(img)
            img = (torch.rand(img.shape, dtype=torch.float32) + img * 255.0) / 256.0

        img = self.transform(img)
        mask = self.get_mask()

        ret = {}
        ret["gt"] = img
        ret["cond"] = img * (1.0 - mask) + mask * torch.randn_like(img)
        ret["mask_image"] = img * (1.0 - mask) + mask
        ret["mask"] = mask

        return ret

    def get_mask(self):
        if self.mask_mode == "manual":
            mask = bbox2mask(self.resolution, self.mask_config["shape"])
        elif self.mask_mode == "fourdirection" or self.mask_mode == "onedirection":
            mask = bbox2mask(
                self.resolution, random_cropping_bbox(mask_mode=self.mask_mode)
            )
        elif self.mask_mode == "hybrid":
            if np.random.randint(0, 2) < 1:
                mask = bbox2mask(
                    self.resolution, random_cropping_bbox(mask_mode="onedirection")
                )
            else:
                mask = bbox2mask(
                    self.resolution, random_cropping_bbox(mask_mode="fourdirection")
                )
        elif self.mask_mode == "file":
            pass
        elif self.mask_mode == "boxAndFreeForm":
            if np.random.randint(0, 2) < 1:
                mask = bbox2mask(
                    self.resolution,
                    random_bbox(
                        img_shape=self.resolution,
                        max_bbox_shape=self.maxBboxMaskSize,
                        max_bbox_delta=self.maxBboxMaskDelta,
                        min_margin=self.minMargin,
                    ),
                )
            else:
                mask = get_irregular_mask(
                    img_shape=self.resolution, area_ratio_range=self.freeFormRatio
                )
        else:
            raise NotImplementedError(
                f"Mask mode {self.mask_mode} has not been implemented."
            )
        return torch.from_numpy(mask).permute(2, 0, 1)


def get_dataloader(
    dataset: tvds.VisionDataset,
    batch_size,
    split: str,
    pin_memory,
    drop_last,
    num_workers,
    distributed,
    uniform_dequantisation=False,
):
    root = ROOT
    if distributed and (split == "train" or split == "sample"):
        batch_size = batch_size // int(os.environ.get("WORLD_SIZE", "1"))

    data_kwargs = {"root": root, "transform": None}
    if dataset == CIFAR10 or dataset == CIFAR10Colourisation:
        data_kwargs["download"] = True
        data_kwargs["train"] = split == "train" or split == "eval"
    else:
        data_kwargs["split"] = "all" if split == "eval" else "train"

    data_kwargs["uniform_dequantisation"] = uniform_dequantisation

    dataset = dataset(**data_kwargs)

    dataloader_configs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "num_workers": num_workers,
    }

    dataloader_configs["sampler"] = sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=drop_last)
        if distributed
        else None
    )
    dataloader_configs["shuffle"] = sampler is None
    dataloader = DataLoader(dataset, **dataloader_configs)

    return dataloader, sampler
