import os
import csv
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

#################### SST Language Data ####################
# SST Data Processor 
class SSTDataProcessor:
    def __init__(self):
        self.lang_dir = load_yaml('paths')['SST_data_dir']

    def train_file(self):
        lang_train_label = []
        lang_train_txt = []
        assert 'train.tsv' in os.listdir(self.lang_dir), 'train.tsv not found in SST data directory'
        with open(os.path.join(self.lang_dir, 'train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                lang_train_label.append(row[1])
                lang_train_txt.append(row[0])

        # convert to pd dataframe
        self.lang_train_df = pd.DataFrame({"labels": lang_train_label, "sentence": lang_train_txt})
        self.lang_train_df = self.lang_train_df.iloc[1:]

    def dev_file(self):
    # development file
        lang_dev_txt = []
        lang_dev_label = []
        assert 'dev.tsv' in os.listdir(self.lang_dir), 'dev.tsv not found in SST data directory'
        with open(os.path.join(self.lang_dir, 'dev.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                lang_dev_txt.append(row[0])
                lang_dev_label.append(row[1])

        # convert to pd dataframe
        self.lang_dev_df = pd.DataFrame({"labels": lang_dev_label, "sentence": lang_dev_txt})
        self.lang_dev_df = self.lang_dev_df.iloc[1:]

    def labels(self):
        # train labels
        lang_train_labels = [int(i) for i in self.lang_train_df['labels']] # convert str to int
        print(f"The number of train labels: {len(lang_train_labels)}")

        # dev labels
        lang_dev_labels = [int(i) for i in self.lang_dev_df['labels']] # convert str to int
        print(f"The number of dev labels: {len(lang_dev_labels)}")

        return lang_train_labels, lang_dev_labels

    def features(self):
        # train features
        vocabs = {} # vocabs
        for sentence in self.lang_train_df["sentence"]:
            for word in sentence.split():
                vocabs[word] = 0

        # train features
        if "lang_train_features.pkl" not in os.listdir(self.lang_dir):
            lang_train_features = []
            for sentence in tqdm(self.lang_train_df["sentence"]):
                bow_train = copy.deepcopy(vocabs)
                for word in sentence.split():
                    if word in bow_train:
                        bow_train[word] += 1
                lang_train_features.append(list(bow_train.values()))
            with open('datasets/SST-2/features/lang_train_features.pkl', 'wb') as f:
                pickle.dump(lang_train_features, f)
        else:
            print("loading train features...")
            with open('datasets/SST-2/features/lang_train_features.pkl', 'rb') as f:
                lang_train_features = pickle.load(f)

        # dev features
        if "lang_dev_features.pkl" not in os.listdir(self.lang_dir):
            lang_dev_features = []
            for sentence in tqdm(self.lang_dev_df["sentence"]):
                bow_dev = copy.deepcopy(vocabs)
                for word in sentence.split():
                    bow_dev = copy.deepcopy(vocabs)
                    if word in bow_dev:
                        bow_dev[word] += 1
                lang_dev_features.append(list(bow_dev.values()))
            with open('datasets/SST-2/features/lang_dev_features.pkl', 'wb') as f:
                pickle.dump(lang_dev_features, f)
        else:
            print("loading dev features...")
            with open('datasets/SST-2/features/lang_dev_features.pkl', 'rb') as f:
                lang_dev_features = pickle.load(f)
        
        return lang_train_features, lang_dev_features

# Custom Language Dataset
class Lang_Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

####################  MNIST Vision Data ####################
# MNIST Data Processor
class MNISTDataProcessor:
    def __init__(self):
        self.vision_dir = load_yaml('paths')['MNIST_data_dir']
    
    def train_file(self):
        self.vision_train = pd.read_csv(os.path.join(self.vision_dir, 'mnist_train.csv'), header=None)
        self.vision_train.rename(columns={0: "labels"}, inplace=True)
    
    def test_file(self):
        self.vision_test = pd.read_csv(os.path.join(self.vision_dir, 'mnist_test.csv'), header=None)
        self.vision_test.rename(columns={0: "labels"}, inplace=True)
    
    def labels(self):
        # labels of vision_train
        vision_train_labels = self.vision_train["labels"]

        # labels of vision_test
        vision_test_labels = self.vision_test["labels"]

        return vision_train_labels, vision_test_labels

    def features(self):
        # normalize each row of vision_train[:, 1:]
        vision_train_features = self.vision_train.iloc[:, 1:].apply(lambda x: (x-np.mean(x))/np.std(x), axis=1)

        # normalize each row of vision_test[:, 1:]
        vision_test_features = self.vision_test.iloc[:, 1:].apply(lambda x: (x-np.mean(x))/np.std(x), axis=1)

        return vision_train_features, vision_test_features

# Custom Vision Dataset
class Vision_Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features.iloc[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
