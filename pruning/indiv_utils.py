import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import os
import copy
import datetime

def load_yaml(filename):
    # LOAD YAML FILE
    with open(f'config/{filename}.yaml','r') as f:
        output = yaml.safe_load(f)
    return output

def get_layers(model):
    # reference: https://saturncloud.io/blog/pytorch-get-all-layers-of-model-a-comprehensive-guide/
    layers = []
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            layers += get_layers(module)
        elif isinstance(module, nn.ModuleList):
            for m in module:
                layers += get_layers(m)
        else:
            layers.append(module)
    return layers


def size_on_disk(model):
    '''
    Reference: https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
    '''
    dir = 'out'
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(), f"{dir}/temp.p")
    size = os.path.getsize(f"{dir}/temp.p")
    print(f"Model Size on Disk: {size/1e6} MB")
    os.remove(f"{dir}/temp.p")
    return size

def print_params_layer(layer: nn.Module) -> None:
    if "Linear" not in layer.__class__.__name__:
        # TODO: Compute the number of params in other types of layers (e.g., Conv2d) 
        print(f"\t{layer.__class__.__name__}: = {0}") # 0 for activation layers
        params = 0
    elif "Linear" in layer.__class__.__name__ and layer.bias is None:
        print(f"\t{layer.__class__.__name__}: {layer.in_features} * {layer.out_features} = {layer.in_features * layer.out_features:,}")
        params = layer.in_features * layer.out_features
    elif "Linear" in layer.__class__.__name__ and layer.bias is not None:
        print(f"\t{layer.__class__.__name__}: {layer.in_features} * {layer.out_features} + {layer.out_features} = {layer.in_features * layer.out_features + layer.out_features:,}")
        params = layer.in_features * layer.out_features + layer.out_features
    return params

def measure_inference_latency(model, test_dataset, device, warmup_itr):
    config = load_yaml('config')
    device = config['device']
    device = torch.device(device)
    print(f"Measuring inference latency of trained {model.__class__.__name__} on {device}...")
    test_dataloader =  DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        inference_latency = []
        for i, data in tqdm(enumerate(test_dataloader)):
            features, _ = data
            features = features.to(device)
            # WARM_UP
            if i == 0:
                print("Warm-up begins...")
                for _ in range(warmup_itr):
                    _ = model(features)
            # MEASURE INFERENCE LATENCY    
            begin = time.time()
            _ = model(features)
            end = time.time()
            inference_latency.append((end - begin)/features.shape[0])
    mean_inference_latency = np.mean(inference_latency)*1000
    print(f"Mean inference latency: {mean_inference_latency:.3f}ms")
    # plot inference latency over iterations and save it as a figure
    if not os.path.exists("out"):
        os.makedirs("out")
    plt.figure(figsize=(12, 8))
    plt.plot(inference_latency)
    plt.scatter(range(len(inference_latency)), inference_latency, s=10, c='r')
    plt.title(f"Inference Latency vs. Iterations in Test Loop\nMean inference latency: {mean_inference_latency:.3f}ms")
    plt.xlabel("Iteration")
    plt.ylabel("Inference latency [s]")
    plt.savefig(f"out/{model.__class__.__name__}_inference_latency.png")
    plt.close()

def param_count(model, layers):
    # COUNT THE NUMBER OF PARAMETERS
    params = 0
    print(f"The number of parameters in each layer of {model.__class__.__name__}:")
    for layer in layers:
        params += print_params_layer(layer)
    print(f"The total number of parameters in {model.__class__.__name__}: {params:,}")

def FLOPs_count(model, layers):
    # COUNT FLOPs (Floating Point Operations) OF LINEAR LAYERS; Consider FMA (Fused Multiply-Add) is used in the hardware architecture.
    fc_layers = [layer for layer in layers if "Linear" in layer.__class__.__name__]
    FLOPs = 0
    for _, fc_layer in enumerate(fc_layers):
        MAC = fc_layer.in_features * fc_layer.out_features # Multiply-Accumulate
        FLOPs += 2 * MAC # 1 FLOP ~= 2 MAC for FMA
    print(f"The total FLOPs in {model.__class__.__name__}: {FLOPs/1e9:.6f} GFLOPs")

def save_model_weights(model, fname):
    if not os.path.exists("out"):
        os.makedirs("out")
    torch.save(model.state_dict(), f"out/{model.__class__.__name__}_weights_{fname}.pth")
    print(f"Saved {model.__class__.__name__}'s weights as out/{model.__class__.__name__}_weights_{fname}.pth.")

def train(model, criterion, optimizer, epochs, train_dataloader, val_dataloader, device):
    val_time = 0.0
    dummy_time = {}

    for epoch in range(epochs):
        # TRAIN
        model.train()
        for i, input_batch in tqdm(enumerate(train_dataloader)):
            # INPUT BATCH: FEATURES and LABELS
            features, labels = input_batch
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
        val_begin = time.time()
        print("Validating...")
        model.eval()
        with torch.no_grad():
            # COMPUTE TRAINING ACCURACY
            train_correct = 0
            for input_batch in train_dataloader:
                features, labels = input_batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, dim=1)
                train_correct += (predicted == labels).sum().item()
                train_acc = train_correct / len(train_dataloader.dataset) * 100
            # COMPUTE VALIDATION ACCURACY
            dev_correct = 0
            for input_batch in val_dataloader:
                features, labels = input_batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, dim=1)
                dev_correct += (predicted == labels).sum().item()
            dev_acc = dev_correct / len(val_dataloader.dataset) * 100
            
        # PRINT STATISTICS
        print(f"Epoch: {epoch+1}/{epochs}, Step: {i+1}/{len(train_dataloader)}, Train Accuracy: {train_acc:.6f}%, Val Accuracy: {dev_acc:.3f}%")
        val_end = time.time()
        val_time += (val_end - val_begin)
    
    dummy_time['val'] = val_time
    return dummy_time

def start_train(model, device, criterion, epochs, batch_size, lr):
    from data_processing import MNISTDataProcessor, Vision_Dataset

    vision_train_features, vision_test_features, vision_train_labels, vision_test_labels = MNISTDataProcessor().features_and_labels()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    vision_train_dataset = Vision_Dataset(vision_train_features, vision_train_labels)
    vision_train_dataloader = DataLoader(vision_train_dataset, batch_size=batch_size, shuffle=True)

    vision_test_dataset = Vision_Dataset(vision_test_features, vision_test_labels)
    vision_test_dataloader = DataLoader(vision_test_dataset, batch_size=batch_size, shuffle=True)

    train(model=model, 
    criterion=criterion,
    optimizer=optimizer, 
    epochs=epochs,
    train_dataloader=vision_train_dataloader, 
    val_dataloader=vision_test_dataloader,
    device=device)