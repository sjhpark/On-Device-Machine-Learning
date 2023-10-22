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
        inference_latency = []
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
            inference_latency.append(end - begin)
    mean_inference_latency = sum(inference_latency) / len(test_dataloader) * 1000
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
            MAC = fc_layer.in_features * fc_layer.out_features # Multiply-Accumulate
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

# Quantization
class Quantize:
    def __init__(self, x_tensor:torch.Tensor, y_type:str):
        '''
        Features:
            Quantization 
                from float to int
            Dequantization
                from int to float
        Input Args:
            x_tensor: input tensor
            y_type: 'int2', 'int8', 'int16', 'int32', 'int64', 'uint2', 'uint8', 'uint16', 'uint32', 'uint64'
        '''
        self.range_dict = {
                    'uint2': (0, 3),
                    'uint8': (0, 255),
                    'uint16': (0, 65535),
                    'uint32': (0, 4294967295),
                    'uint64': (0, 18446744073709551615),

                    'int2': (-2, 1),
                    'int8': (-128, 127),
                    'int16': (-32768, 32767),
                    'int32': (-2147483648, 2147483647),
                    'int64': (-9223372036854775808, 9223372036854775807),
                    } 

        self.tensor = x_tensor # input tensor
        self.y_type = y_type # target type
        
        self.q_min, self.q_max = self.range_dict[self.y_type]
        self.x_min, self.x_max = torch.min(self.tensor), torch.max(self.tensor)
    
    def scale_and_zero_point(self, x_min:float, x_max:float, q_min, q_max) -> Tuple[float, int]:
        # scale factor
        s = (x_max - x_min) / (q_max - q_min)

        # zero point
        if torch.round(x_min / s) == q_min:
            z = 0
        else:
            z = -torch.round(x_min / s) + q_min
        
        return s, z
    
    def linear_mapping(self):
        '''
        Quantization
        Q(r) = round(r / s) + z
        '''
        self.s, self.z = self.scale_and_zero_point(self.x_min, self.x_max, self.q_min, self.q_max)
        tensor = torch.round(self.tensor/self.s) + self.z
        tensor = torch.clamp(tensor, self.q_min, self.q_max)
        return tensor # quantized tensor

    def inverse_mapping(self):
        '''
        Dequantization
        r' = (Q(r) - z) * s
        '''
        tensor = self.linear_mapping()
        tensor = (tensor - self.z) * self.s
        return tensor # dequantized tensor
    
    def error(self):
        '''
        Quantization Error
        error = |r - r'|
        '''
        tensor = self.inverse_mapping()
        tensor = torch.abs(self.tensor - tensor)
        return tensor
