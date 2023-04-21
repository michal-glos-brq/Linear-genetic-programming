"""
This Python3 module implements a wrapper around torchvision or custom created dummy datasets.

Dataset class provides basic methods and properties to easily download or create the dataset
as well as normalize and split data into training and evaluation sets. 
"""

import sys
from typing import Callable, Union, Tuple, Iterable, Dict
from numbers import Number

import numpy as np
from tqdm import tqdm
from pprint import pformat
from PIL import Image

import torch
from torchvision import datasets
from torchvision.transforms import PILToTensor


def resizePIL(edge_size: Union[int, None]) -> Callable[[Image.Image], torch.Tensor]:
    """
    Function factory for: square PIL image resizer + to tensor converters

    Args:
        edge_size Union[int, None]: Image edge size after resizing, None if no resizing
    Returns:
        Callable[[PIL.Image.Image], torch.Tensor]: function resizing \
            PIL images and converting them into torch.Tensor
    """
    # Instantiate PIL Image to torch Tensor converter
    pil_to_tensor = PILToTensor()

    def inner_fn(img):
        """Resize if required, then convert to torch tensor"""
        # Resize
        if edge_size:
            img = img.resize((edge_size, edge_size))
        # Convert to torch Tensor
        nonlocal pil_to_tensor
        return pil_to_tensor(img)

    # Factory output
    return inner_fn


class Dataset:
    """
    Simple dataset wrapper, either torchvision or custom dataset with one-hot encoded labels as data

    Provides basic functions - Downloading or creating dataset, splitting datasetto test and eval,
        normalizing dataset or providing dataset metadata.

    Attributes:
        classes (List[str]): Subscriptable data structure containing class names.
        X (torch.Tensor): Dataset tensor (data).
        y (torch.Tensor): Dataset tensor (labels).
        device (torch.device): Torch device (GPU if available, otherwise CPU).
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
        # Obtain torch device (Try to obtain GPU, fallback to CPU if GPU not found)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # If test dataset not required, use the chosen torch dataset
        if test is None:
            self.get_torch_dataset(name, root, edge_size)
        else:
            self.get_test_dataset(*test)

        self.edge_size = edge_size if edge_size else "orig"

        # Apply data normalization
        self.normalize_data(*normalize)

        # Apply equation to compute boundaries on self.data for training|eval data (choose min - limit|actual_size)
        self.dataset_split = int(data_split / 100.0 * self.X.shape[0])

    def get_torch_dataset(self, name: str, root: str, edge_size: Union[None, int]) -> None:
        """
        Obtain chosen dataset from torchvision module

        Args:
            name (str): Dataset name.
            root (str): Dataset folder (downloads end up in there).
            edge_size (Union[None, int]): Size of square of dataset images to be resized to (None when no resize).
        """
        # Save the dataset loaded
        self.dataset_name = name
        # Check for dataset name, exit the app if incorrect dataset required
        if hasattr(datasets, name):
            dataset_cls = getattr(datasets, name)
        else:
            sys.exit(f"Dataset with name {name} does not exist in torchivision datasets")

        # Obtain training and testing data
        train = dataset_cls(root=root, train=True, download=True, transform=resizePIL(edge_size))
        test = dataset_cls(root=root, train=False, download=True, transform=resizePIL(edge_size))

        # Merge both into a single dataset containing all data
        data, labels = [], []
        for img, label in tqdm(train + test, ncols=100, desc="Loading dataset ..."):
            data.append(img)
            labels.append(label)

        # Obtain a list of class names
        self.classes = np.array(train.classes)

        # Prepare data - merge test and train data into single tensor
        with tqdm(total=2, desc="Preparing dataset ...", ncols=100) as pbar:
            # Load and convert labels into Tensors
            self.y = torch.tensor(labels).to(self.device)
            pbar.update(1)
            # Load and convert data into Tensors
            self.X = torch.cat(data).to(self.device)
            pbar.update(1)

    def get_test_dataset(self, classes: int, entries: int) -> None:
        """
        Create simple test dataset. Generate random dataset labels.
        From labels create one-hot encoded vectors, which would be used as data.

        Args:
            classes (int): Number of classes (np.arange(classes) are labels).
            entries (int): Number of data entries to generate.
        """
        # Save the dataset loaded
        self.dataset_name = "Test/Dummy"
        with tqdm(total=3, desc="Creating dataset ...", ncols=100) as pbar:
            # Generate radom classes labels
            self.y = torch.randint(0, classes, (entries,)).to(self.device)
            pbar.update(1)
            # For each label, generate the same object
            self.X = torch.zeros((entries, classes)).to(self.device)
            pbar.update(1)
            self.X = self.X.scatter(1, self.y.unsqueeze(1), 1).to(self.device)
            pbar.update(1)

        # Labels (subscriptable structure contains class names)
        self.classes = np.arange(classes).astype(str)

    def normalize_data(self, lower_bound: Number, upper_bound: Number) -> None:
        """
        Linear normalization of data into <lower, upper> interval

        Args:
            lower_bound:    Normalization lower bound
            upper_bound:    Normalization upper bound
        """
        # Save norm bounds in order to be displayd
        self.normalized_lower, self.normalized_upper = lower_bound, upper_bound
        # 1. Normalize into interval <0,1>
        self.X = self.X / (self.X.max() - self.X.min())
        # 2. Normalize into required interval span
        self.X = (self.X * (upper_bound - lower_bound)) + lower_bound

    @property
    def class_count(self) -> int:
        """Return number of classes in dataset"""
        return len(self.classes)

    @property
    def obj_shape(self) -> Iterable[int]:
        """Return the shape of objects from dataset"""
        return self.X.shape[1:]

    @property
    def test_X(self) -> torch.Tensor:
        """Obtain testing data (objects)"""
        return self.X[: self.dataset_split]

    @property
    def test_y(self) -> torch.Tensor:
        """Obtain testing labels (classes)"""
        return self.y[: self.dataset_split]

    @property
    def eval_X(self) -> torch.Tensor:
        """Obtain eval data (objects)"""
        return self.X[self.dataset_split :]

    @property
    def eval_y(self) -> torch.Tensor:
        """Obtain eval labels (classes)"""
        return self.y[self.dataset_split :]

    @property
    def _info_dict(self) -> Dict[str, Dict]:
        """Obtain dictionary with information about self"""
        return {
            "Dataset": {
                "entries training": self.dataset_split,
                "entries eval": len(self.X) - self.dataset_split,
                "entries total": len(self.X),
                "device": self.device,
                "name": self.dataset,
            },
            "Labels": {
                "distinct labels": self.class_count,
                "label names": self.classes,
            },
            "Objects": {
                "shape": self.obj_shape,
                "norm. upper": self.normalized_upper,
                "norm. upper real": self.X.max(),
                "norm. lower": self.normalized_lower,
                "norm. lower real": self.X.min(),
            },
        }

    def __repr__(self) -> str:
        """Return string representing of information about self"""
        return pformat(self._info_dict, width=120)
