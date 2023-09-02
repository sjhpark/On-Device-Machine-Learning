import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# yaml loader
def load_yaml(filename):
    with open(f'config/{filename}.yaml','r') as f:
        output = yaml.safe_load(f)
    return output

def train(model, criterion, optimizer, epochs, train_dataloader, dev_dataloader, device):
    for epoch in range(epochs):

        # TRAIN
        model.train()
        for i, data in tqdm(enumerate(train_dataloader)):
            # INPUT BATCH: FEATURES and LABELS
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            # ZERO OUT THE GRADIENTS
            optimizer.zero_grad()
            # FORWARD PASS
            outputs = model(features)
            # LOSS COMPUTATION
            loss = criterion(outputs, labels)
            # BACKWARD PASS
            loss.backward()
            # WEIGHTS UPDATE
            optimizer.step()
            # ACCURACY COMPUTATION
            if i % (len(train_dataloader)//10) == 0: # every 10% of an epoch
                model.eval()
                with torch.no_grad():
                    train_correct = 0
                    for data in train_dataloader:
                        features, labels = data
                        features = features.to(device)
                        labels = labels.to(device)
                        outputs = model(features)
                        _, predicted = torch.max(outputs.data, dim=1)
                        train_correct += (predicted == labels).sum().item()
                        train_acc = train_correct / len(train_dataloader.dataset) * 100
                    # EVAL
                    dev_correct = 0
                    for data in dev_dataloader:
                        features, labels = data
                        features = features.to(device)
                        labels = labels.to(device)
                        outputs = model(features)
                        _, predicted = torch.max(outputs.data, dim=1)
                        dev_correct += (predicted == labels).sum().item()
                    dev_acc = dev_correct / len(dev_dataloader.dataset) * 100
                    # PRINT STATISTICS
                    print(f"Epoch: {epoch+1}/{epochs}, Step: {i+1}/{len(train_dataloader)}, Train Accuracy: {train_acc:.3f}%, Eval Accuracy: {dev_acc:.3f}%")

    