import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
from torch.utils.data import Dataset

from data import LABEL_NAMES


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            return correct_k.mul_(100.0 / batch_size).item()


def train(model: nn.Module, criterion: nn.Module, data: Dataset, epochs: int,
          device: torch.device) -> float:
    """
    Training loop for model

    Args:
        model (nn.Module): Neural net
        criterion (nn.Module): Loss function
        data (Dataset): Training data
        epochs (int): Number of epochs
        device (torch.device): Device to train model

    Returns:
        float: fitness value according to accuracy
    """
    sample, _ = list(data)[0]
    summary(model.cuda(), input_size=sample.shape[1:])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())

    fitness = 0
    for epoch in range(epochs):
        print('Epoch: ', epoch + 1)
        train_loss = 0
        highest_acc = 0
        for batch, (img, target) in enumerate(data):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            loss = loss.item() * img.size(0)
            train_loss += loss
            acc = accuracy(pred, target)
            highest_acc = acc if acc > highest_acc else highest_acc
            #print(
            #    f'Batch: {batch} of {len(data)}\tAcc: {acc}\tLoss: {round(loss, 2)}'
            #)
        fitness = highest_acc if highest_acc > fitness else fitness
        print(
            f'Epoch: {epoch+1} of {epochs}\tAcc: {highest_acc}\tLoss: {round(train_loss/len(data), 2)}'
        )

    return fitness
