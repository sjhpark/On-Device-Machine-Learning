import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

def print_params_layer(layer: nn.Module, parmas_dict: dict) -> None:
    # TODO print the number of params in other types of layers (e.g., Conv2d)
    if isinstance(layer, nn.Linear) == False:
        print(f"\t{layer.__class__.__name__}: = {parmas_dict[layer.__class__.__name__]:,}")
    elif isinstance(layer, nn.Linear) and layer.bias is None:
        print(f"\t{layer.__class__.__name__}: {layer.in_features} * {layer.out_features} = {parmas_dict[layer.__class__.__name__]:,}")
    elif isinstance(layer, nn.Linear) and layer.bias is not None:
        print(f"\t{layer.__class__.__name__}: {layer.in_features} * {layer.out_features} + {layer.out_features} = {parmas_dict[layer.__class__.__name__]:,}")

def measure_inference_latency_CPU(model, test_dataset, device, warmup_itr):
    print(f"Measuring inference latency of trained {model.__class__.__name__} on CPU...")
    test_dataloader =  DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        inference_latency = 0
        for i, data in tqdm(enumerate(test_dataloader)):
            features, _ = data
            features = features.to(device)
            # WARM_UP
            if i == 0:
                print("Warm-up begins...")
                for _ in range(warmup_itr):
                    _ = model(features)
                print("Warm-up complete!")
            # MEASURE INFERENCE LATENCY    
            begin = time.time()
            _ = model(features)
            end = time.time()
            inference_latency += (end - begin)
    mean_inference_latency = inference_latency / len(test_dataloader) * 1000
    print(f"Mean inference latency: {mean_inference_latency:.3f}ms")

def benchmarking(func):
    def wrapper(*args, **kwargs):
        device = kwargs['device']
        model = kwargs['model'].to(device)
        test_dataset = kwargs['test_dataset']
        layers = get_layers(model)

        # COUNT THE NUMBER OF PARAMETERS
        parmas_dict = {}
        print(f"The number of parameters in each layer of {model.__class__.__name__}:")
        for layer in layers:
            parmas_dict[layer.__class__.__name__] = sum(params.numel() for params in layer.parameters())
            print_params_layer(layer, parmas_dict)
        num_params = sum(params.numel() for params in model.parameters())
        print(f"The total number of parameters in {model.__class__.__name__}: {num_params:,}")
        
        # COUNT FLOPs (Floating Point Operations) OF LINEAR LAYERS; Consider FMA (Fused Multiply-Add) is used in the hardware architecture.
        fc_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        FLOPs = 0
        for _, fc_layer in enumerate(fc_layers):
            MAC = fc_layer.in_features * fc_layer.out_features # Multiply-Accumulate (*2 b/c Muliplication & Addition to Accumulator)
            FLOPs += 2 * MAC # 1 FLOP ~= 2 MAC for FMA
        print(f"The total FLOPs in {model.__class__.__name__}: {FLOPs/1e9:.6f} GFLOPs")

        # COUNT INFERENCE LATENCY
        warmup_itr = 100
        measure_inference_latency_CPU(model, test_dataset, device, warmup_itr)

        # COUNT TOTAL TRAINING TIME
        begin = time.time()
        dummy_time = func(*args, **kwargs)
        end = time.time()
        print(f"Total training time: {(end - begin) - sum(dummy_time.values()):.3f}s")
        print(f"Dummy times that have been excluded from training: {dummy_time}")
    return wrapper

@benchmarking
def train(model, criterion, optimizer, epochs, train_dataloader, val_dataloader, test_dataset, device, val):
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
            # --INFERENCE TIME ENDS--
            end = time.time()
            # LOSS COMPUTATION
            loss = criterion(outputs, labels)
            # BACKWARD PASS
            loss.backward()
            # WEIGHTS UPDATE
            optimizer.step()

        # ACCURACY COMPUTATION
        if val:
            val_begin = time.time()
            print("Validating...")
            model.eval()
            with torch.no_grad():
                train_correct = 0
                for input_batch in train_dataloader:
                    features, labels = input_batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, dim=1)
                    train_correct += (predicted == labels).sum().item()
                    train_acc = train_correct / len(train_dataloader.dataset) * 100
                # EVAL
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
