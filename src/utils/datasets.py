"""
This Python3 module implements a wrapper around torchvision or custom created dummy datasets.

Dataset class provides basic methods and properties to easily download or create the dataset
as well as normalize and split data into training and evaluation sets.
"""

from pprint import pformat
from functools import partial
from typing import Callable, Union, Tuple, Dict
from numbers import Number

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor


def resize_and_convert_pil(edge_size: Union[int, None]) -> Callable[[Image.Image], torch.Tensor]:
    """
    Function factory for: square PIL image resizer + to tensor converter
    All resized images are squares

    Args:
        edge_size Union[int, None]: Image edge size after resizing, None if no resizing
    Returns:
        Callable[[PIL.Image.Image], torch.Tensor]: function resizing \
            PIL images and converting them into torch.Tensor
    """
    # Instantiate PIL Image to torch Tensor converter
    to_tensor = ToTensor()

    # Create a partial function for resizing images if edge_size is provided
    resize_fn = partial(Image.Image.resize, size=(edge_size, edge_size)) if edge_size else None

    def process_image(img):
        """
        Resize the image (if required) and convert it to a torch tensor.

        Args:
            img (Image.Image): Input PIL image.

        Returns:
            torch.Tensor: Output torch tensor.
        """
        if edge_size:
            img = resize_fn(img)
        return to_tensor(img)

    return process_image

# pylint: disable=invalid-name, too-many-arguments
class Dataset:
    """
    Simple dataset wrapper, either torchvision or custom dataset with one-hot encoded labels as data

    Provides basic functions - Downloading or creating dataset, splitting datasetto test and eval,
        normalizing dataset or providing dataset metadata.

    Attributes:
        classes (List[str]): List with class names.
        X (torch.Tensor): Dataset tensor (data).
        y (torch.Tensor): Dataset tensor (labels).
        torch_device (torch.torch_device): Torch torch_device (GPU if available, otherwise CPU).
    """

    def __init__(
        self,
        name: str = "MNIST",
        root: str = "data",
        edge_size: Union[int, None] = None,
        data_split: int = 10,
        normalize: Tuple[Number] = (0, 1),
        test: Union[Tuple[Number], None] = None,
    ) -> None:
        """
        Check args and obtain/load required dataset with provided configuration.

        Args:
            name (str): Dataset name (var ignored when test dataset chosen).
            root (str): Root folder to store downloaded datasets.
            edge_size (Union[int, None]): Resize images to squares of edge_size size (square datasets used only).
            data_split (int): Percentage of dataset used for training, the rest is used for evaluating.
            normalize (Tuple[Number]): Iterable of length > 1, interval for linear normalization of image pixels.
            test (Union[Tuple[Number], None]): Tuple with 2 numbers: (number of classes, number of entries in dataset).
        """
        self.torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if test is None:
            self.load_torch_dataset(name, root, edge_size)
        else:
            self.create_test_dataset(*test)

        # For naming purpose only
        self.edge_size = str(edge_size) if edge_size else "orig"

        with torch.no_grad():
            self.normalize_data(*normalize)

        self.dataset_split = int(data_split / 100.0 * self.X.size(0))
        self.class_count = len(self.classes)
        self.object_shape = self.X.shape[1:]
        self.test_X = self.X[:self.dataset_split]
        self.test_y = self.y[:self.dataset_split]
        self.eval_X = self.X[self.dataset_split:]
        self.eval_y = self.y[self.dataset_split:]


    def load_torch_dataset(self, name: str, root: str, edge_size: Union[None, int]) -> None:
        """
        Obtain chosen dataset from torchvision module

        Args:
            name (str): Dataset name.
            root (str): Dataset folder (downloads end up in there).
            edge_size (Union[None, int]): Size of square of dataset images to be resized to (None when no resize).
        """
        # Obtain test and train torchvision datasets
        self.dataset_name = name
        dataset_cls = getattr(datasets, name)

        train = dataset_cls(root=root, train=True, download=True, transform=resize_and_convert_pil(edge_size))
        test = dataset_cls(root=root, train=False, download=True, transform=resize_and_convert_pil(edge_size))
        self.classes = train.classes

        # Merge both datasets
        data, labels = [], []
        for img, label in tqdm(train + test, ncols=100, desc="Loading dataset ..."):
            data.append(img)
            labels.append(label)

        with tqdm(total=2, desc="Preparing dataset ...", ncols=100) as pbar:
            self.y = torch.tensor(labels).to(self.torch_device)
            pbar.update(1)

            if data[0].size(0) == 1:
                self.X = torch.cat(data).to(self.torch_device)
            else:
                self.X = torch.stack(data).to(self.torch_device)
            pbar.update(1)


    def create_test_dataset(self, classes: int, entries: int) -> None:
        """
        Create simple test dataset. Generate random dataset labels.
        From labels create one-hot encoded vectors, which would be used as data.

        Args:
            classes (int): Number of classes.
            entries (int): Number of data entries to generate.
        """
        self.dataset_name = "Test-dataset"
        with tqdm(total=3, desc="Creating dataset ...", ncols=100) as pbar:
            self.y = torch.randint(0, classes, (entries,), device=self.torch_device)
            pbar.update(1)
            self.X = torch.zeros((entries, classes), device=self.torch_device)
            pbar.update(1)
            self.X = self.X.scatter(1, self.y.unsqueeze(1), 1)
            pbar.update(1)
        self.classes = list(range(1, classes + 1))

    def normalize_data(self, lower_bound: Number, upper_bound: Number) -> None:
        """
        Linear normalization of data into <lower, upper> interval

        Args:
            lower_bound:    Normalization lower bound
            upper_bound:    Normalization upper bound
        """
        self.normalized_lower, self.normalized_upper = lower_bound, upper_bound
        self.X = self.X / (self.X.max() - self.X.min())
        self.X = (self.X * (upper_bound - lower_bound)) + lower_bound


    @property
    def _info_dict(self) -> Dict[str, Dict]:
        """Provide crucial Information about dataset as a dictionary"""
        return {
            "Dataset": {
                "entries training": self.dataset_split,
                "entries eval": len(self.X) - self.dataset_split,
                "entries total": len(self.X),
                "torch_device": self.torch_device,
                "name": self.dataset_name,
            },
            "Labels": {
                "distinct labels": self.class_count,
                "label names": self.classes,
            },
            "Objects": {
                "shape": self.object_shape,
                "norm. upper": self.normalized_upper,
                "norm. upper real": self.X.max(),
                "norm. lower": self.normalized_lower,
                "norm. lower real": self.X.min(),
            },
        }

    def __repr__(self) -> str:
        """Provide string representation of Dataset class instance"""
        return pformat(self._info_dict, width=120)
