from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import config
from dataset import CarControlDataset
from model import SimpleCNN


def main():
    device = torch.device('cuda:0')

    dataset = CarControlDataset(Path('data/first_dataset'))
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    image_x, image_y = config.get('camera.resolution').values()
    model = SimpleCNN(image_x, image_y).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        losses = []
        for i, data in enumerate(dataset_loader):
            x, y_hat = data
            x = x.to(device)
            y_hat = y_hat.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y_hat)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f'epoch {epoch} avg loss: {sum(losses) / len(losses)}')


if __name__ == '__main__':
    main()