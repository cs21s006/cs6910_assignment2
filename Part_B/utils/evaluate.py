import torch
from torch import nn
from tqdm import tqdm
torch.manual_seed(7)
import torch.nn.functional as F


def evaluate(device, loader, model, model_name):
    ''' Function to calculate accuracy to see performance of our model '''
    correct_samples = 0
    total_samples = 0
    model.eval()

    loss = .0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in tqdm(loader, total=len(loader)):
            x = x.to(device=device)
            y = y.to(device=device)

            outputs = model(x)
            outputs = F.softmax(outputs, dim=1)

            loss += criterion(outputs, y).item()

            _, predictions = outputs.max(1)
            correct_samples += (predictions == y).sum().item()
            total_samples += predictions.size(0)

    acc = round((correct_samples / total_samples) * 100, 4)
    return acc, loss/len(loader)
