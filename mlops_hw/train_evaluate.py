import torch
import torch.nn.functional as F
from torchmetrics import F1Score


def train_model(model, epochs, optimizer, loader, device):
    model.train()
    loss = None
    acc = None
    f1score = F1Score("multiclass", num_classes=10)
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            _, out = torch.max(out.data, 1)
            acc = (out == target).sum().item()
            f1 = f1score(out, target)
            loss.backward()
            optimizer.step()

    return loss.item(), acc, f1


def evaluate_model(model, loader, device):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_sum += F.cross_entropy(out, target).item()
            _, out = torch.max(out.data, 1)
            acc_sum += (out == target).sum().item()

        loss_sum /= len(loader)
        acc_sum /= len(loader.dataset)
    return loss_sum, acc_sum
