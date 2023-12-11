from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from dataset import CarControlDataset, DatasetSplit
from loss import WeightedMSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model import DrivingModel

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
    steer_criterion: nn.Module,
    steer_weight: float,
    throttle_criterion: nn.Module,
    throttle_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    outputs = model(x)

    steer_loss = steer_criterion(outputs[:, 0], y[:, 0])
    throttle_loss = throttle_criterion(outputs[:, 1], y[:, 1])
    total_loss = steer_weight * steer_loss + throttle_weight * throttle_loss

    total_loss.backward()
    optimizer.step()
    return total_loss, steer_loss, throttle_loss


def test_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    steer_criterion: nn.Module,
    steer_weight: float,
    throttle_criterion: nn.Module,
    throttle_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.to(device)
    y = y.to(device)

    outputs = model(x)

    steer_loss = steer_criterion(outputs[:, 0], y[:, 0])
    throttle_loss = throttle_criterion(outputs[:, 1], y[:, 1])
    total_loss = steer_weight * steer_loss + throttle_weight * throttle_loss

    return total_loss, steer_loss, throttle_loss


def main():
    learning_rate = config.get('training.learning_rate')
    batch_size = config.get('training.batch_size')
    epochs = config.get('training.epochs')
    dataset_path = Path(config.get('dataset.folder_path'))
    save_path = Path(config.get('training.save_path'))
    image_x, image_y = config.get('camera.resolution').values()

    train_dataset = CarControlDataset(dataset_path, DatasetSplit.TRAIN)
    test_dataset = CarControlDataset(dataset_path, DatasetSplit.TEST)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = DrivingModel().to(device)

    steer_criterion = nn.MSELoss()
    steer_weight = 1.0
    throttle_criterion = nn.MSELoss()
    throttle_weight = 0.5
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # wandb.watch(model, criterion, log='all', log_freq=100, log_graph=True)

    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch}')

        model.train()
        train_losses: dict[str, list[float]] = {
            'total_loss': [],
            'steer_loss': [],
            'throttle_loss': [],
        }
        for data in tqdm(train_loader):
            x, y = data
            total_loss, steer_loss, throttle_loss = train_batch(
                x,
                y,
                model,
                optimizer,
                steer_criterion,
                steer_weight,
                throttle_criterion,
                throttle_weight,
            )
            train_losses['total_loss'].append(total_loss.item())
            train_losses['steer_loss'].append(steer_loss.item())
            train_losses['throttle_loss'].append(throttle_loss.item())

            wandb.log(
                {
                    'train/total_loss': total_loss.item(),
                    'train/steer_loss': steer_loss.item(),
                    'train/throttle_loss': throttle_loss.item(),
                    'epoch': epoch,
                }
            )

        model.eval()
        test_losses = {
            'total_loss': [],
            'steer_loss': [],
            'throttle_loss': [],
        }
        with torch.no_grad():
            for data in tqdm(test_loader):
                x, y = data
                total_loss, steer_loss, throttle_loss = test_batch(
                    x,
                    y,
                    model,
                    steer_criterion,
                    steer_weight,
                    throttle_criterion,
                    throttle_weight,
                )
                test_losses['total_loss'].append(total_loss.item())
                test_losses['steer_loss'].append(steer_loss.item())
                test_losses['throttle_loss'].append(throttle_loss.item())

        train_mean_losses = {key: sum(losses) / len(losses) for key, losses in train_losses.items()}
        test_losses = {key: sum(losses) / len(losses) for key, losses in test_losses.items()}
        wandb.log(
            {
                'test/total_loss': test_losses['total_loss'],
                'test/steer_loss': test_losses['steer_loss'],
                'test/throttle_loss': test_losses['throttle_loss'],
                'train/mean_total_loss': train_mean_losses['total_loss'],
                'train/mean_steer_loss': train_mean_losses['steer_loss'],
                'train/mean_throttle_loss': train_mean_losses['throttle_loss'],
                'epoch': epoch,
            }
        )
        print(f'{train_mean_losses = }')
        print(f'{test_losses = }\n')

        # TODO: save model with wandb and clean up this mess :)
        model_path = save_path / f'epoch_{epoch}.pl'
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path.as_posix())


if __name__ == '__main__':
    main()
