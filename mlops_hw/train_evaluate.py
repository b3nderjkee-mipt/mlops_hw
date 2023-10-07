import torch
import torch.nn.functional as F


def train(model, optimizer, loader, device):
    model.train()
    for idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
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
