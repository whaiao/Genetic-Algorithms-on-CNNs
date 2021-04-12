import torch
import torch.nn as nn
import torch.nn.functional as F

from data import LABEL_NAMES


def train(model: nn.Module, criterion, data, epochs, device):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.001,
                                momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=200)
    train_loss = 0
    correct = 0
    total = 0

    fitness = 0
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        for batch, (img, target) in enumerate(data):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = round(100. * correct / total, 2)

            print(
                f'Batch: {batch} of {len(data)}\tAcc: {acc}\tLoss: {train_loss/(batch+1)}'
            )
        fitness = acc if acc > fitness else fitness
        print(
            f'Epoch: {epoch+1} of {len(epochs)+1}\tAcc: {acc}\tLoss: {round(train_loss/(batch+1))}'
        )

    return fitness
