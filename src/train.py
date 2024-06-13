import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from config import config
from driving_model import DrivingModel
from loss import PartialL1Loss, PartialL2Loss
from torch_datasets import DirectControlDataset, WaypointPredictionDataset, WaypointSamplingMethod

device = torch.device('cuda:0')
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
random.seed(0)

wandb.init(
    project='naive-driving-agent-v0',
    config=config.config,
    mode=None if config['training.use_wandb'] else 'disabled',
)

# TODO: Create a better permanent solution for paramters from sweep overwriting config from config.yaml
config.config['training'] = {**config.config['training'], **wandb.config}

current_step = 0


def process_batch(
    imgs: torch.Tensor,
    target: torch.Tensor,
    nav_tps: torch.Tensor,
    nav_commands: torch.Tensor,
    model: nn.Module,
    criterions: Sequence[tuple[str, float, nn.Module]],
    optimizer: optim.Optimizer | None = None,
    include_nav_tps: bool = True,
    include_nav_commands: bool = True,
) -> dict[str, float]:
    is_training = optimizer is not None
    global current_step
    with torch.set_grad_enabled(is_training):
        imgs = imgs.to(device)
        target = target.to(device)
        nav_tps = nav_tps.to(device)
        nav_commands = nav_commands.to(device)

        model.train(mode=is_training)
        if optimizer:
            optimizer.zero_grad()

        outputs = model(
            imgs,
            nav_tps if include_nav_tps else None,
            nav_commands if include_nav_commands else None,
        )

        all_losses = {}
        weighted_losses = []
        for loss_name, weight, criterion in criterions:
            loss = criterion(outputs, target)
            assert not torch.isnan(loss), f'loss is nan: {loss_name}'
            all_losses[loss_name] = loss.item()
            weighted_losses.append(weight * loss)
        total_weighted_loss = sum(weighted_losses)

        if optimizer:
            total_weighted_loss.backward()
            optimizer.step()
            current_step += 1

        all_losses['total_loss'] = total_weighted_loss.item()
        return all_losses


def process_epoch(
    loader,
    model,
    criterions,
    optimizer=None,
    log_batch=False,
    log_prefix='',
    include_nav_tps: bool = True,
    include_nav_commands: bool = True,
):
    loss_sums: dict[str, float] = defaultdict(lambda: 0.0)
    for imgs, nav_tps, nav_commands, target in tqdm(loader):
        losses = process_batch(
            imgs,
            target,
            nav_tps,
            nav_commands,
            model,
            criterions,
            optimizer,
            include_nav_tps,
            include_nav_commands,
        )

        for loss_name, loss_value in losses.items():
            loss_sums[loss_name] += target.shape[0] * loss_value

        if log_batch:
            batch_metrics = {}
            for loss_name, loss_value in losses.items():
                batch_metrics[f'{log_prefix}/{loss_name}/batch'] = loss_value
                wandb.log(batch_metrics, current_step)

    epoch_metrics = {}
    for loss_name, loss_sum in loss_sums.items():
        epoch_metrics[f'{log_prefix}/{loss_name}/total'] = loss_sum / len(loader.dataset)
    wandb.log(epoch_metrics, current_step)
    print(epoch_metrics)
    return epoch_metrics


def save_model(model: nn.Module, epoch: int, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / f'model_epoch_{epoch}.pt'
    torch.save(model.state_dict(), model_path.as_posix())

    wandb.log_artifact(model_path, name=f'model_epoch_{epoch}', type='model')


def main():
    print(f'Config: {config}')

    dataset_path = Path(config['dataset.folder_path'])
    train_dataset_path = dataset_path / config['dataset.train_sub_path']
    val_dataset_path = dataset_path / config['dataset.val_sub_path']
    train_excluded_towns = config['dataset.train_excluded_towns']
    image_x, image_y = config['camera.resolution'].values()
    initial_lr = config['training.initial_lr']
    lr_decay_factor= config['training.lr_decay_factor']
    weight_decay = config['training.weight_decay']
    batch_size = config['training.batch_size']
    epochs = config['training.epochs']
    vision_bottleneck = config['training.vision_bottleneck']
    vision_bottleneck_capacity = config['training.vision_bottleneck_capacity']
    ignore_vision_prob = config['training.ignore_vision_prob']
    ignore_vision_feature_vector_zeros = config['training.ignore_vision_feature_vector_zeros']
    ignore_vision_feature_vector_random = config['training.ignore_vision_feature_vector_random']
    totally_ignore_vision = config['model.totally_ignore_vision']
    totally_ignore_tp = config['model.totally_ignore_tp']
    save_path = Path(config['training.save_path'])
    prediction_type = config['model.predict.type']

    jitter = config['training.color_jitter']
    transform = transforms.Compose(
        (
            transforms.ColorJitter(
                jitter['brightness'], jitter['contrast'], jitter['saturation'], jitter['hue']
            ),
        )
    )

    if prediction_type == 'waypoints':
        num_waypoints = config['model.predict.num_waypoints']
        sampling_interval = config['model.predict.waypoint_sampling_interval']
        sampling_method = WaypointSamplingMethod(config['model.predict.sampling_method'])
        model = DrivingModel(
            2 * num_waypoints,
            vision_bottleneck,
            vision_bottleneck_capacity,
            ignore_vision_prob,
            ignore_vision_feature_vector_zeros,
            ignore_vision_feature_vector_random,
            totally_ignore_vision,
            totally_ignore_tp,
        ).to(device)
        criterions = []
        for i in range(num_waypoints):
            distance = sampling_interval * (i + 1)
            criterions.append(
                (f'waypoint_{distance}m_L1_loss', 1.0 / distance, PartialL1Loss(2 * i, 2 * i + 2))
            )
        train_dataset = WaypointPredictionDataset(
            train_dataset_path,
            num_waypoints,
            sampling_interval,
            sampling_method,
            transform,
            train_excluded_towns,
        )
        val_dataset = WaypointPredictionDataset(
            val_dataset_path, num_waypoints, sampling_interval, sampling_method
        )
    elif prediction_type == 'direct_controls':
        steer_loss_weight = config['model.predict.steer_loss_weight']
        throttle_loss_weight = config['model.predict.throttle_loss_weight']
        brake_loss_weight = config['model.predict.brake_loss_weight']
        model = DrivingModel(
            3,
            vision_bottleneck,
            vision_bottleneck_capacity,
            ignore_vision_prob,
            ignore_vision_feature_vector_zeros,
            ignore_vision_feature_vector_random,
            totally_ignore_vision,
            totally_ignore_tp,
        ).to(device)
        criterions = (
            ('steer_loss', steer_loss_weight, PartialL2Loss(0, 1)),
            ('throttle_loss', throttle_loss_weight, PartialL2Loss(1, 2)),
            ('brake_loss', brake_loss_weight, PartialL2Loss(2, 3)),
        )
        train_dataset = DirectControlDataset(train_dataset_path, transform, train_excluded_towns)
        val_dataset = DirectControlDataset(val_dataset_path)
    else:
        raise ValueError(f'{prediction_type} is not a valid value for model.predict.type')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), initial_lr, weight_decay=weight_decay)
    lr_scheduler = ExponentialLR(optimizer, lr_decay_factor)

    for epoch in range(epochs):
        print(f'epoch {epoch}')

        process_epoch(
            train_loader,
            model,
            criterions,
            optimizer,
            log_batch=True,
            log_prefix='train',
            include_nav_tps=True,
            include_nav_commands=False,
        )
        process_epoch(
            val_loader,
            model,
            criterions,
            log_prefix='val',
            include_nav_tps=True,
            include_nav_commands=False,
        )

        lr_scheduler.step()
        wandb.log({'epoch': epoch, 'lr': lr_scheduler.get_last_lr()[0]}, current_step)

        save_model(model, epoch, save_path)


if __name__ == '__main__':
    main()
