import torch.nn.functional as F
from torch.autograd import Variable


def train(model, optimizer, loader):
    model.train()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_sum += loss.data[0]
        loss.backward()
        optimizer.step()

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum()
        acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)


def evaluate(model, loader):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_sum += loss.data[0]

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum()
        acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)
