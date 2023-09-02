import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import torch

def load_yaml(filename):
    # LOAD YAML FILE
    with open(f'config/{filename}.yaml','r') as f:
        output = yaml.safe_load(f)
    return output

def compute_time(func):
    def timer(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"Time taken for running {func.__name__}: {(end - begin):.3f}s")
    return timer

def computer_inference_latency_GPU(model, dummy_features, device, iterations=100):
    assert device == torch.device("cuda"), "This method is valid only for GPU as your device."
    assert iterations >= 100 and iterations <= 1000, "Iterations should be greater than 100 and less than 1000."
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_array = np.zeros((iterations, 1)) # preallocate an array of size of iterations

    # WARM-UP
    print("Warming up...")
    for _ in range(iterations//10):
        _ = model(dummy_features)
    print ("Warming up complete")
    
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

@compute_time
def train(model, criterion, optimizer, epochs, train_dataloader, dev_dataloader, device):
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

        # ACCURACY COMPUTATION
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

        # MEAN INFERENCE TIME PER INPUT DATA PER EPOCH
        mean_time = total_time / inputs_count * 1000 # in [ms]
        print(f"Mean inference time per input data for epoch {epoch}: {mean_time:.6f}ms")


    