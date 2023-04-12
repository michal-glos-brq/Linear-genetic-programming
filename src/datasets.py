                           #######################
                           #@$%&             &%$@#
                           #!    Michal Glos    !#
                           #!     EVO - 2023    !#
                           #!        __         !#
                           #!      <(o )___     !#
                           #!       ( ._> /     !#
                           #@$%&     `---'   &%$@#
                           #######################


'''
This python3 module implements all dataset-related functionality for this project
'''

import sys

from pprint import pformat
from tqdm import tqdm
import numpy as np
import torch

from torchvision import datasets
from torchvision.transforms import PILToTensor

def resizePIL(edge_size):
    '''
    Function factory for square PIL images resizers + to tensor converters
    
    @args:
        edge_size - Required edge size of img (None if resizing not required)
    '''
    def inner_fn(img):
        '''Resize if required, convert to torch tensor'''
        # Resize
        if edge_size:
            img = img.resize((edge_size, edge_size))
        # Convert to torch Tensor
        return PILToTensor()(img)

    # Factory output
    return inner_fn


class Dataset:
    '''
    Dataset class - torchvision dataset wrapper or dummy test dataset
        with one-hot encoded vectors of labels as data

    important properties:
        classes     - subscriptable data structure (self.classes[class_id] holds value for class name)
        X           - dataset tensor (data)
        y           - dateset tensor (labels)
        device      - torch computational device (GPU if found)
    '''    

    def __init__(self, name='MNIST', root='data', edge_size=None, data_split=10,
                 normalize=(0,1), test=None):
        '''
        Check args and obtain/load required dataset with provided configuration
        
        @args:
            name:           Dataset name (var ignored when test dataset chosen)
            root:           Root folder to store downloaded datasets
            edge_size:      Resize images to squares of edge_size size (sqare datasets used only)
            data_split:     Percentage of dataset used for training, the rest is used for evaluating
            normalize:      Iterable of lenght > 1, interval for linear normalization of image pixels
            test:           Use test (dummy one hot vectors as data) dataset
        '''
        # Obtain torch device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # If test dataset not required, use the chosen torch dataset
        if test is None:
            self.get_torch_dataset(name, root, edge_size)
        else:
            self.get_test_dataset(*test)

        # Apply data normalization
        self.normalize_data(*normalize)

        # Establish and save dataset boundaries for testing|eval
        assert 0 < data_split < 100, (f'0 < {data_split} < 100 is False, choose splitpoint correclty (in %).')
        # Apply equation to compute boundaries on self.data for training|eval data (choose min - limit|actual_size)
        self.dataset_split = int(data_split / 100. * self.X.shape[0])


    def get_torch_dataset(self, name, root, edge_size):
        '''
        Obtain chosen dataset from torchvision module
        
        @args:
            name:           Dataset name
            root:           Dataset folder (downloads end up in there)
            edge_size:      Resize dataset images to edge_size x edge_size
        '''
        # Save the dataset loaded
        self.dataset = name
        # Check for dataset name, exit the app if incorrect dataset required
        if hasattr(datasets, name):
            dataset_cls = getattr(datasets, name)
        else:
            sys.exit(f'Dataset with name {name} does not exist in torchivision datasets')

        # Obtain training and testing data
        train = dataset_cls(root=root, train=True, download=True, transform=resizePIL(edge_size))
        test = dataset_cls(root=root, train=False, download=True, transform=resizePIL(edge_size))

        # Merge both into a single dataset containing all data
        data, labels = [], []
        for img, label in tqdm(train + test, ncols=100, desc='Loading dataset ...'):
            data.append(img)
            labels.append(label)

        # Obtain a list of class names
        self.classes = train.classes

        # Prepare data - merge test and train data into single tensor
        with tqdm(total=2, desc='Preparing dataset ...', ncols=100) as pbar:
            # Load and convert labels into Tensors
            self.y = torch.tensor(labels).to(self.device); pbar.update(1)
            # Load and convert data into Tensors
            self.X = torch.cat(data).to(self.device); pbar.update(1)


    def get_test_dataset(self, classes, entries):
        '''
        Create very easy testing dataset consisting of one-hot encoded vectors as data
        Dataset size is derived from entries limit

        @args:
            classes:        Number of classes (np.arange(classes) are labels then)
            entries:        Number of data entries to generate
        '''
        # Save the dataset loaded
        self.dataset = 'Test/Dummy'
        with tqdm(total=3, desc='Creating dataset ...', ncols=100) as pbar:
            # Generate radom classes labels
            self.y = torch.randint(0, classes, (entries,)).to(self.device); pbar.update(1)
            # For each label, generate the same object
            self.X = torch.zeros((entries, classes)).to(self.device); pbar.update(1)
            self.X = self.X.scatter(1, self.y.unsqueeze(1), 1).to(self.device); pbar.update(1)

        # Labels (subscriptable structure contains class names)
        self.classes = np.arange(classes).astype(str)


    def normalize_data(self, lower_bound, upper_bound):
        '''
        Linear normalization of data into <lower, upper> interval
        
        @args:
            lower_bound:    Normalization lower bound
            upper_bound:    Normalization upper bound
        '''
        # Save norm bounds in order to be displayd
        self.normalized_lower, self.normalized_upper = lower_bound, upper_bound
        with torch.no_grad():
            # 1. Normalize into interval <0,1>
            self.X = self.X / (self.X.max() - self.X.min())
            # 2. Normalize into required interval span
            self.X = (self.X * (upper_bound - lower_bound)) + lower_bound

    @property
    def class_count(self):
        '''How many classes are there in dataset?'''
        return len(self.classes)

    @property
    def obj_shape(self):
        '''Return shape of data objects'''
        return self.X.shape[1:]

    @property
    def test_X(self):
        '''Obtain X data (objects) used for testing'''
        return self.X[ : self.dataset_split]
    
    @property
    def test_y(self):
        '''Obtain y data (labels) used for testing'''
        return self.y[ : self.dataset_split]
    
    @property
    def eval_X(self):
        '''Obtain X data (objects) used for evaluation'''
        return self.X[self.dataset_split : ]
    
    @property
    def eval_y(self):
        '''Obtain y data (labels) used for evaluation'''
        return self.y[self.dataset_split : ]

    def _info_dict(self):
        '''Obtain dictionary with information about self'''
        return {
            'Object shape': self.obj_shape,
            'Objects training': self.dataset_split,
            'Objects eval': len(self.X) - self.dataset_split,
            'Objects total': len(self.X),
            'Distinct classes': self.class_count,
            'Object min value': self.X.min(),
            'Object max value': self.X.max(),
            'Classes': ', '.join(self.classes),
            'Normalization upper bound': self.normalized_upper,
            'Normalization lower bound': self.normalized_lower,
            'Device': self.device,
            'Dataset': self.dataset
        }

    def __repr__(self):
        '''Return string representing of information about self'''
        return pformat(self._info_dict(), width=120)
