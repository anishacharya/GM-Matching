from torchvision import datasets, transforms
import numpy as np
import torch


def get_data(
    dataset_name, 
    split="train",
    transform=None,
    download=True, 
    data_dir="./data"
):

    """
    Load datasets for either vision tasks or LLM benchmarks.

    Parameters:
        dataset_name (str):
        split (str):
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
                                        Only applies for vision tasks.
        download (bool): Whether to download the dataset if it is not already present.
        data_dir (str): Directory where the data is (or will be) stored.

    Returns:
        dataset: A dataset object.
    """
    dataset = None

    if dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        n_classes = 10

        if split == "train":
            if transform is None:
                # Default transforms for training data
                tr_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),

                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    
                    Cutout(n_holes=1, length=16)
                ])
            else:
                tr_transform = transform

            dataset = datasets.CIFAR10(
                root=data_dir,
                train=True,
                transform=tr_transform,
                download=download
            )

        elif split == "test":
            if transform is None:
                # Default transforms for test data
                te_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            else:
                te_transform = transform

            dataset = datasets.CIFAR10(
                root=data_dir,
                train=False,
                transform=te_transform,
                download=download
            )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    return dataset, n_classes


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

