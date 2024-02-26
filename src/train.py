from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from episode import DatasetSplit
from loss import PartialL2Loss
from model import DrivingModel
from torch_datasets import DirectControlDataset, WaypointPredictionDataset

device = torch.device('cuda:0')
wandb.init(
    project='naive-driving-agent-v0',
    config={
        'learning_rate': config['training.learning_rate'],
        'batch_size': config['training.batch_size'],
        'prediction_task': 'wp',
    },
    mode=None if config['training.use_wandb'] else 'disabled',
)

# TODO: Create a better permanent solution for paramters from sweep overwriting config from config.yaml
config.config['training'] = {**config.config['training'], **wandb.config}


def process_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    criterions: Sequence[tuple[str, float, nn.Module]],
    optimizer: optim.Optimizer | None = None,
) -> dict[str, float]:
    x = x.to(device)
    y = y.to(device)

    # Choose train/eval mode depending on presence of optimizer
    model.train(mode=optimizer is not None)
    if optimizer:
        optimizer.zero_grad()

    outputs = model(x)

    all_losses = {}
    weighted_losses = []
    for loss_name, weight, criterion in criterions:
        loss = criterion(outputs, y)
        all_losses[loss_name] = loss.item()
        weighted_losses.append(weight * loss)
    total_weighted_loss = sum(weighted_losses)

    if optimizer:
        total_weighted_loss.backward()
        optimizer.step()

    all_losses['total_loss'] = total_weighted_loss.item()
    return all_losses


def save_model(model: nn.Module, epoch: int, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / f'model_epoch_{epoch}.pt'
    torch.save(model.state_dict(), model_path.as_posix())

    wandb.log_artifact(model_path, name=f'model_epoch_{epoch}', type='model')


def main():
    dataset_path = Path(config['dataset.folder_path'])
    image_x, image_y = config['camera.resolution'].values()
    learning_rate = config['training.learning_rate']
    batch_size = config['training.batch_size']
    epochs = config['training.epochs']
    save_path = Path(config['training.save_path'])
    prediction_type = config['model.predict.type']

    if prediction_type == 'waypoints':
        num_waypoints = config['model.predict.num_waypoints']
        sampling_interval = config['model.predict.waypoint_sampling_interval']
        model = DrivingModel(out_size=3 * num_waypoints).to(device)
        criterions = []
        for i in range(num_waypoints):
            distance = sampling_interval * (i + 1)
            criterions.append((f'waypoint_{distance}m_L2_loss', 1.0, PartialL2Loss(i, i + 3)))
        dataset_args = (
            dataset_path,
            num_waypoints,
            sampling_interval,
        )
        train_dataset = WaypointPredictionDataset(*dataset_args, split=DatasetSplit.TRAIN)
        val_dataset = WaypointPredictionDataset(*dataset_args, split=DatasetSplit.VAL)
        test_dataset = WaypointPredictionDataset(*dataset_args, split=DatasetSplit.TEST)
    elif prediction_type == 'direct_controls':
        steer_loss_weight = config['model.predict.steer_loss_weight']
        throttle_loss_weight = config['model.predict.throttle_loss_weight']
        brake_loss_weight = config['model.predict.brake_loss_weight']
        model = DrivingModel(out_size=3).to(device)
        criterions = (
            ('steer_loss', steer_loss_weight, PartialL2Loss(0, 1)),
            ('throttle_loss', throttle_loss_weight, PartialL2Loss(1, 2)),
            ('brake_loss', brake_loss_weight, PartialL2Loss(2, 3)),
        )
        train_dataset = DirectControlDataset(dataset_path, DatasetSplit.TRAIN)
        val_dataset = DirectControlDataset(dataset_path, DatasetSplit.VAL)
        test_dataset = DirectControlDataset(dataset_path, DatasetSplit.TEST)
    else:
        raise ValueError(f'{prediction_type} is not a valid value for model.predict.type.')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = optim.Adam(model.parameters(), learning_rate)

    for epoch in range(epochs):
        print(f'epoch {epoch}')

        train_loss_sums: dict[str, float] = defaultdict(lambda: 0.0)
        for x, y in tqdm(train_loader):
            losses = process_batch(
                x,
                y,
                model,
                criterions,
                optimizer,
            )

            log_dict = {}
            log_dict['epoch'] = epoch
            for loss_name, loss_value in losses.items():
                log_dict[f'train/{loss_name}'] = loss_value
                train_loss_sums[loss_name] += y.shape[0] * loss_value
            wandb.log(log_dict)

        val_loss_sums: dict[str, float] = defaultdict(lambda: 0.0)
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                losses = process_batch(
                    x,
                    y,
                    model,
                    criterions,
                )
                for loss_name, loss_value in losses.items():
                    val_loss_sums[loss_name] += y.shape[0] * loss_value

        test_loss_sums: dict[str, float] = defaultdict(lambda: 0.0)
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                losses = process_batch(
                    x,
                    y,
                    model,
                    criterions,
                )
                for loss_name, loss_value in losses.items():
                    test_loss_sums[loss_name] += y.shape[0] * loss_value

        log_dict = {}
        log_dict['epoch'] = epoch
        for loss_name, loss_sum in train_loss_sums.items():
            log_dict[f'train/mean_{loss_name}'] = loss_sum / len(train_dataset)
        for loss_name, loss_sum in val_loss_sums.items():
            log_dict[f'val/{loss_name}'] = loss_sum / len(val_dataset)
        for loss_name, loss_sum in test_loss_sums.items():
            log_dict[f'test/{loss_name}'] = loss_sum / len(test_dataset)

        wandb.log(log_dict)
        print(f'{log_dict = }\n')

        save_model(model, epoch, save_path)


if __name__ == '__main__':
    main()
