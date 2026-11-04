import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import numpy as np
import glob
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from time import time

import torchvision

class SUN397Dataset(Dataset):
    """
    A custom dataset class for loading the SUN397 dataset.
    """

    def populate_dataset(self):
        assert self.data_dir is not None, "what happened?"
        top_dirs = sorted(os.listdir(self.data_dir))
        labels = []

        for dir in top_dirs:
            dir_path = os.path.join(self.data_dir, dir)
            if os.path.isfile(dir_path):
                continue

            sorted_dirs = sorted(os.listdir(dir_path))
            curr_labels = [os.path.join(dir, s) for s in sorted_dirs if s[0] != '.'] 
            
            labels.extend(curr_labels)

        self.data = []
        for i, dir in enumerate(labels):
            curr_dir_path = os.path.join(self.data_dir, dir)
            files = os.listdir(curr_dir_path)
            for f in files:
                item = ( os.path.abspath(os.path.join(curr_dir_path, f)), torch.tensor(i) )
                self.data.append(item)

    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset with images and labels.
        Args:
            data_dir (str): Path to the data directory.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir
        self.data = None
        self.transform = transform

        self.populate_dataset()
    
    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        assert self.data is not None, "how is it not populated at __len__?"

        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an image and its label at the specified index.
        
        Args:
            idx (int): Index of the image to retrieve.
        
        Returns:
            tuple: (image, label)
        """
        assert self.data is not None, "how is self.data not populated in __getitem__?"

        img_path, label = self.data[idx]
        img = Image.open(img_path)
        img = img.convert("RGB")
        arr = np.array(img)

        if self.transform is not None and self.transform is callable:
            arr = self.transform(arr)

        return (transforms.ToTensor().__call__(arr), label)



class CNN(nn.Module):
    """
    Define your CNN Model here 
    """
    def __init__(self, num_classes=10):
        """
        Initializes the layers of the CNN model.
        
        Args:
            num_classes (int): Number of output classes.
        """
        pass

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output of the model.
        """
        pass


def offline_mean_std(dataloader: DataLoader):
    assert dataloader.batch_size == 1, "offline_mean_std expects batch_size = 1"

    total_r, total_g, total_b, = 0.0, 0.0, 0.0
    total2_r, total2_g, total2_b, = 0.0, 0.0, 0.0

    for image, _ in dataloader:
        ...

    pass

    
def calculate_mean_std(**kwargs):
    """
    Fill in the per channel mean and standard deviation of the dataset. 
    Just fill in the values, no need to compute them.
    """
    # return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]

    return [0, 0 ,0], [0,0,0]
    pass

'''
All of the following functions are optional. They are provided to help you get started.
'''

def train(model, train_loader, **kwargs):
    pass

def test(model, test_loader, **kwargs):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, 
                        default='welcome/to/CNN/homework',
                        help='Path to training data directory')
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
 
    return parser.parse_args()
    
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    data = SUN397Dataset('hw4-data')

if __name__ == "__main__":
    main()
