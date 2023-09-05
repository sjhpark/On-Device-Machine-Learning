import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import torch
import torch.nn as nn

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

def benchmarking(func):
    def wrapper(*args, **kwargs):
        model = kwargs['model']
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

        # COUNT TRAINING TIME
        begin = time.time()
        dummy_time = func(*args, **kwargs)
        end = time.time()
        print(f"Dummy times that have been excluded from {func.__name__}: {dummy_time}")
        print(f"Total time taken for running {func.__name__}: {(end - begin) - sum(dummy_time.values()):.3f}s")
    return wrapper

def computer_inference_latency_GPU(model, dummy_features, device, iterations=100):
    assert device == torch.device("cuda"), "This method is valid only for GPU as your device."
    assert iterations >= 100 and iterations <= 1000, "Iterations should be greater than 100 and less than 1000."
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_array = np.zeros((iterations, 1)) # preallocate an array of size of iterations

    # WARM-UP
    print("Warming up...")
    for _ in range(iterations//10):
        _ = model(dummy_features)
    print ("Warming up complete.")
    
    with torch.no_grad():
        for itr in range(iterations):
            begin.record()
            _ = model(dummy_features)
            end.record()
            torch.cuda.synchronize()
            time = begin.elapsed_time(end)
            time_array[itr] = time # store the time

    mean_time = np.sum(time_array) / iterations # mean inference time
    print(f"Mean inference time: {mean_time:.3f}ms")

@benchmarking
def train(model, criterion, optimizer, epochs, train_dataloader, dev_dataloader, device, eval, warmup_itr=100):
    assert warmup_itr >= 100 and warmup_itr <= 1000, "Iterations should be greater than 100 and less than 1000."
    dummy_time = {}

    for epoch in range(epochs):
        inputs_count = 0
        total_time = 0.0 # total inference time
        # TRAIN
        model.train()
        for i, data in tqdm(enumerate(train_dataloader)):
            # INPUT BATCH: FEATURES and LABELS
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            batch_size = features.shape[0]
            # ZERO OUT THE GRADIENTS
            optimizer.zero_grad()
            # --Warm-up--
            being_warmup = time.time()
            if epoch == 0 and i == 0:
                print("Warm-up begins...")
                for _ in range(warmup_itr):
                    _ = model(features)
                end_warmup = time.time()
                dummy_time['warmup'] = (end_warmup - being_warmup)
                print("Warm-up complete!")
            # --INFERENCE TIME BEGINS--
            begin = time.time()
            # FORWARD PASS
            outputs = model(features)
            # --INFERENCE TIME ENDS--
            end = time.time()
            # --COMPUTE BATCH INFERENCE TIME--
            batch_time = end - begin
            total_time += batch_time
            inputs_count += batch_size
            # LOSS COMPUTATION
            loss = criterion(outputs, labels)
            # BACKWARD PASS
            loss.backward()
            # WEIGHTS UPDATE
            optimizer.step()
        
        # MEAN INFERENCE TIME PER INPUT DATA PER EPOCH
        mean_time = total_time / inputs_count * 1000 # in [ms]
        print(f"Mean inference time per input data for epoch {epoch}: {mean_time:.6f}ms")

        # ACCURACY COMPUTATION PER EPOCH
        eval_time = 0.0
        if eval:
            eval_begin = time.time()
            print("Evaluating...")
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
            print(f"Epoch: {epoch+1}/{epochs}, Step: {i+1}/{len(train_dataloader)}, Train Accuracy: {train_acc:.6f}%, Eval Accuracy: {dev_acc:.3f}%")
            eval_end = time.time()
            eval_time += (eval_end - eval_begin)
    
    dummy_time['eval'] = eval_time
    return dummy_time
