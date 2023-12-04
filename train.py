from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config import config
from dataset import CarControlDataset, get_train_test_car_control_datasets
from loss import WeightedMSELoss
from model import DrivingModel, SimpleCNN

device = torch.device('cuda:0')
wandb.init(
    project='naive-driving-agent-v0',
    config={
        'learning_rate': config.get('training.learning_rate'),
        'batch_size': config.get('training.batch_size'),
    },
)


def train_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
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
    model: nn.Module,
    criterion: nn.Module,
):
    x = x.to(device)
    y = y.to(device)

    outputs = model(x)

    loss = criterion(outputs, y)
    return loss


def main():
    num_train_episodes = config.get('training.num_train_episodes')
    num_test_episodes = config.get('training.num_test_episodes')
    learning_rate = config.get('training.learning_rate')
    batch_size = config.get('training.batch_size')
    epochs = config.get('training.epochs')
    save_path = config.get('training.save_path')
    image_x, image_y = config.get('camera.resolution').values()

    train_dataset, test_dataset = get_train_test_car_control_datasets(
        Path('data/first_dataset'), num_train_episodes, num_test_episodes
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = DrivingModel().to(device)

    criterion = WeightedMSELoss(weights=[1, 0.5, 0], device=device)
    optimizer = optim.Adam(model.parameters(), learning_rate)

    wandb.watch(model, criterion, log='all', log_freq=1, log_graph=True)

    for epoch in range(1, epochs + 1):
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
                wandb.log({'batch_train_loss': loss.item(), 'epoch': epoch})

        avg_train_loss = sum(train_losses) / len(train_losses)
        test_loss = sum(test_losses) / len(test_losses)
        print(f'{avg_train_loss = :.5f}')
        print(f'{test_loss = :.5f}')
        print()
        wandb.log({'epoch': epoch, 'avg_train_loss': avg_train_loss, 'test_loss': test_loss})

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
