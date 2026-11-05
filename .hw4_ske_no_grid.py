import os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import argparse
import numpy as np
import glob
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from time import time

import torchvision

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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
        self.num_classes = len(labels)
        
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
        super().__init__()

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

        assert self.transform is not None, "transform was empty or wrong"

        return (self.transform(img), label)

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
        super().__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        def calculate_fc_input():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.pool(self.conv3(self.pool(self.conv2(self.pool(self.conv1(dummy))))))
            return out.shape

        _, out_channels, h, w = calculate_fc_input()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=out_channels, out_features=num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output of the model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)

        return self.fc(x)

"""
mean std        = [0.5127, 0.4529, 0.3974], [0.2496, 0.2534, 0.2615]
train mean std  = [0.5144, 0.4533, 0.3984], [0.2503, 0.2540, 0.2621]
val mean std    = [0.5106, 0.4516, 0.3943], [0.2471, 0.2515, 0.2593]
"""

def offline_mean_std(dataloader: DataLoader):
    assert dataloader.batch_size == 1, "offline_mean_std expects batch_size = 1"

    total  = torch.zeros(3, dtype=torch.float64)
    total2 = torch.zeros(3, dtype=torch.float64)

    num_pixels = 0

    for image, _ in dataloader:
        image = image.double()

        squared = image ** 2
        total  += image.sum(axis=(0,2,3))
        total2 += squared.sum(axis=(0,2,3))

        h, w = image.shape[-2:]
        num_pixels += h * w

    mean_r, mean_g, mean_b = total / num_pixels
    mean2_r, mean2_g, mean2_b = total2 / num_pixels

    std_r = torch.sqrt(mean2_r - mean_r**2)
    std_g = torch.sqrt(mean2_g - mean_g**2)
    std_b = torch.sqrt(mean2_b - mean_b**2)

    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]

def train_mean_std():
    return 

def calculate_mean_std(**kwargs):
    """
    Fill in the per channel mean and standard deviation of the dataset. 
    Just fill in the values, no need to compute them.
    """
    # return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]
    return [0.5127, 0.4529, 0.3974], [0.2496, 0.2534, 0.2615]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5127, 0.4529, 0.3974], std=[0.2496, 0.2534, 0.2615])
])

'''
All of the following functions are optional. They are provided to help you get started.
'''

def train(model: CNN, train_loader: DataLoader, val_loader: DataLoader, **kwargs):
    model.to(device=device)
    model.train()

    num_epochs = 25
    lr, wd = kwargs['learning_rate'], kwargs['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_model = None

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn()
    ) as progress:
        config_task = progress.add_task(f"[bold]Config:[/bold] lr={lr}, wd={wd}", total=1)
        epoch_task = progress.add_task("Epochs", total=num_epochs)

        for epoch in range(num_epochs):
            batch_task = progress.add_task(f"Epoch {epoch+1} batches", total=len(train_loader))

            running_loss = 0
            total_images = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                total_images += images.size(0)

                progress.update(batch_task, advance=1)

            progress.remove_task(batch_task)

            val_accuracy = test(model, val_loader)
            model.train()

            epoch_loss = running_loss / total_images
            progress.update(
                epoch_task, 
                advance=1,
                description=f"Epoch {epoch+1}/{num_epochs} | loss={epoch_loss:.4f}. | val_acc{val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
                progress.console.print(f"[green]New best val accuracy: {val_accuracy:.4f} (epoch {epoch+1}) -- recorded")

        progress.update(config_task, completed=1)

    return best_model, best_val_accuracy



def test(model: CNN, test_loader: DataLoader):
    model.to(device=device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def grid_search(train_loader: DataLoader, val_loader: DataLoader, num_classes: int):

    learning_rates = [5e-4]
    weight_decays = [0]

    param_grid = [{'learning_rate': lr, 
                   'weight_decay': wd}  for lr in learning_rates 
                                        for wd in weight_decays]

    best_model = None
    best_hyperparameters = None
    best_accuracy = 0.0

    for param in param_grid:
        model = CNN(num_classes=num_classes)
        curr_model, curr_accuracy = train(model, train_loader, val_loader, **param)

        if curr_accuracy > best_accuracy:
            best_accuracy = curr_accuracy
            best_model = curr_model
            best_hyperparameters = param

    assert best_model is not None and best_hyperparameters is not None, "grid_search"
    return best_model, best_hyperparameters, best_accuracy

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
    
    data = SUN397Dataset('hw4-data', transform)
    train_data, val_data = random_split(data, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=32)

    best_model, best_hyperparameters, best_accuracy = grid_search(train_loader, val_loader, data.num_classes)

    best_model.cpu()
    torch.save(best_model.state_dict(), 'submissions/model.pt')

    print(f"best hyperparameters: {best_hyperparameters}")
    print(f"best_accuracy: {best_accuracy}")

if __name__ == "__main__":
    main()
