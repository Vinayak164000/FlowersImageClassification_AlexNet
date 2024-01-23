import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as Optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import AlexNet
from train import Trainer

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
data_path = Path("C:\\Users\\hp\\OneDrive\\Desktop\\computervision\\flowers")
train_dir = data_path / 'train'
test_dir = data_path / 'test'

if __name__ == "__main__":
    # Load data and create data loader
    data_transform = transforms.Compose([
        transforms.Resize(size=(227, 227)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=data_transform,
                                      target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle=False,
        drop_last=True,
    )
    # ...

    # Create model, loss function, and optimizer
    model = AlexNet(num_classes=5)
    # model = AlexNet(num_classes=len(train_data.classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    # Create trainer and train the model
    trainer = Trainer(model, train_loader, test_loader, optimizer, loss_fn=loss_fn)
    trainer.train()
    # torch.save(model.state_dict(), "C:\\Users\\hp\\OneDrive\\Desktop\\computervision\\flowers\\model_architecture.pth")
