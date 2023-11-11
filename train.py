from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from dataset import CarControlDataset, get_train_test_car_control_datasets
from model import SimpleCNN

device = torch.device('cuda:0')


def train_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    model: SimpleCNN,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
):
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    outputs = model(x)

    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss


def test_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    model: SimpleCNN,
    criterion: nn.Module,
):
    x = x.to(device)
    y = y.to(device)

    outputs = model(x)

    loss = criterion(outputs, y)
    return loss


def main():
    train_dataset, test_dataset = get_train_test_car_control_datasets(
        Path('data/first_dataset'), num_train_episodes=120, num_test_episodes=10
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)

    image_x, image_y = config.get('camera.resolution').values()
    model = SimpleCNN(image_x, image_y).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        print(f'epoch {epoch}')

        model.train()
        train_losses = []
        for data in tqdm(train_loader):
            x, y = data
            loss = train_batch(x, y, model, optimizer, criterion)
            train_losses.append(loss.item())

        model.eval()
        test_losses = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                x, y = data
                loss = test_batch(x, y, model, criterion)
                test_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        test_loss = sum(test_losses) / len(test_losses)
        print(f'{train_loss = }')
        print(f'{test_loss = }')
        print()

    save_path = config.get('training.save_path')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
