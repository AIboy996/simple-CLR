import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms, io
from typing import Literal


class ViewGenerator:
    """Take some random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class CIFAR100(Dataset):
    def __init__(
        self,
        dataset_dir,
        transform=None,
        subset: Literal["train", "test"] = "train",
        device="cpu",
    ) -> None:
        self.data_dict = self.unpickle(os.path.join(dataset_dir, subset))
        self.data = self.data_dict[b"data"]
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        img = self.data.__getitem__(index)

        # resize test

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.from_numpy(img).view(3,32,32).permute(1,2,0))
        # plt.savefig('./tmp/cifar_raw.png')

        img = torch.from_numpy(img).to(device=self.device).view(3, 32, 32).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    @staticmethod
    def unpickle(file, encoding="bytes"):
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding=encoding)
        return dict


class ImageNet200(Dataset):
    def __init__(
        self,
        dataset_dir,
        transform=None,
        subset: Literal["train", "test", "valid"] = "train",
        device="cpu",
    ) -> None:
        # load data without labels
        self.data_dict = {"data": self.loadimages(os.path.join(dataset_dir, subset))}
        self.data = self.data_dict["data"]
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        img_path = self.data.__getitem__(index)
        img = io.read_image(img_path).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    @staticmethod
    def loadimages(path, pattern="**/*.JPEG"):
        return list(Path(path).glob(pattern))


class CLRDataset:

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # https://arxiv.org/abs/2002.05709

        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * size)),
                # transforms.RandomRotation(degrees=90),
            ]
        )
        return data_transforms

    @classmethod
    def get_dataset(self, name, n_views, device="cpu"):
        datasets_available = {
            "cifar100-train": CIFAR100(
                dataset_dir="./data/cifar-100-python",
                subset="train",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
            "cifar100-test": CIFAR100(
                dataset_dir="./data/cifar-100-python",
                subset="test",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
            "imagenet200-train": ImageNet200(
                dataset_dir="./data/tiny-imagenet-200",
                subset="train",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
            "imagenet200-test": ImageNet200(
                dataset_dir="./data/tiny-imagenet-200",
                subset="test",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
            "imagenet200-valid": ImageNet200(
                dataset_dir="./data/tiny-imagenet-200",
                subset="val",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
        }

        try:
            dataset = datasets_available[name]
        except KeyError:
            raise ValueError("Invalid dataset selection.")
        else:
            return dataset


if __name__ == "__main__":
    dataset = CLRDataset.get_dataset(name="cifar100-train", n_views=2, device="cuda:2")
    x = dataset[0]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(x[0][0].permute(1, 2, 0).cpu().numpy())
    axes[1].imshow(x[1][0].permute(1, 2, 0).cpu().numpy())
    fig.savefig("./tmp/cifar_augmented.png")

    dataset = CLRDataset.get_dataset(
        name="imagenet200-valid", n_views=2, device="cuda:2"
    )
    x = dataset[0]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(x[0][0].permute(1, 2, 0).cpu().numpy())
    axes[1].imshow(x[1][0].permute(1, 2, 0).cpu().numpy())
    fig.savefig("./tmp/imagenet_augmented.png")
