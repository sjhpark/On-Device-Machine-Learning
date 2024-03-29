import os
import csv
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from arguments import arguments

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchtext
from torchtext.data import get_tokenizer

# configs
config = load_yaml('config')

#################### SST Language Data ####################
# SST Data Processor 
class SSTDataProcessor:
    def __init__(self):
        self.lang_dir = config['paths']['SST_data_dir']
        self.tokenizer = get_tokenizer('basic_english')
        self.lang_train_df, self.vocabs = self.train_file()
        self.lang_dev_df = self.dev_file()

    def train_file(self):
        lang_train_label = []
        lang_train_txt = []
        vocabs = []
        assert 'train.tsv' in os.listdir(self.lang_dir), 'train.tsv not found in SST data directory'
        with open(os.path.join(self.lang_dir, 'train.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                lang_train_label.append(row[1])
                lang_train_txt.append(row[0])
                vocabs += self.tokenizer(row[0])

        # convert to pd dataframe
        lang_train_df = pd.DataFrame({"labels": lang_train_label, "sentence": lang_train_txt})
        lang_train_df = lang_train_df.iloc[1:]
        vocabs = {word: 0 for word in list(set(vocabs))}
        return lang_train_df, vocabs

    def dev_file(self):
    # development file
        lang_dev_label = []
        lang_dev_txt = []
        assert 'dev.tsv' in os.listdir(self.lang_dir), 'dev.tsv not found in SST data directory'
        with open(os.path.join(self.lang_dir, 'dev.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                lang_dev_txt.append(row[0])
                lang_dev_label.append(row[1])

        # convert to pd dataframe
        lang_dev_df = pd.DataFrame({"labels": lang_dev_label, "sentence": lang_dev_txt})
        lang_dev_df = lang_dev_df.iloc[1:]
        return lang_dev_df

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
        if "lang_train_features.pkl" not in os.listdir("out/"):
            print("saving train features as a pkl file...")
            lang_train_features = []
            for sentence in tqdm(self.lang_train_df["sentence"]):
                vocabs = copy.deepcopy(self.vocabs)
                for word in self.tokenizer(sentence):
                    if word in vocabs:
                        vocabs[word] += 1
                lang_train_features.append(list(vocabs.values()))
            with open('out/lang_train_features.pkl', 'wb') as f:
                pickle.dump(lang_train_features, f)
        else:
            print("parsing train features...")
            with open('out/lang_train_features.pkl', 'rb') as f:
                lang_train_features = pickle.load(f)

        # dev features
        if "lang_dev_features.pkl" not in os.listdir("out/"):
            print("saving dev features as a pkl file...")
            lang_dev_features = []
            for sentence in tqdm(self.lang_dev_df["sentence"]):
                vocabs = copy.deepcopy(self.vocabs)
                for word in self.tokenizer(sentence):
                    if word in vocabs:
                        vocabs[word] += 1
                lang_dev_features.append(list(vocabs.values()))
            with open('out/lang_dev_features.pkl', 'wb') as f:
                pickle.dump(lang_dev_features, f)
        else:
            print("parsing dev features...")
            with open('out/lang_dev_features.pkl', 'rb') as f:
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
        self.vision_dir = config['paths']['MNIST_data_dir']

        self.transform_flag = "True"
        self.transform_type = "Resize"
        self.resize_ratio = 1 # the number to divide the original image size by

        self.vision_train = self.train_file()
        self.vision_test = self.test_file()

    def transform(self, transform_type, resize_ratio, dataset):
        # Resize images (by Resize or CenterCrop)
        image_size = dataset.iloc[:, 1:].values[0].shape
        original_shape = (1, int(image_size[0]**0.5), int(image_size[0]**0.5))
        new_image_size = int((image_size[0]**0.5)//resize_ratio)
        
        if transform_type == 'Resize':
            print(f"Resizing test images from {int(image_size[0]**0.5)}x{int(image_size[0]**0.5)} to {new_image_size}x{new_image_size}")
            transform = transforms.Resize((new_image_size, new_image_size), antialias=True)
        elif transform_type == "Crop":
            print(f"Cropping images from {int(image_size[0]**0.5)}x{int(image_size[0]**0.5)} to {new_image_size}x{new_image_size}")
            transform = transforms.CenterCrop((new_image_size, new_image_size))
        
        image1D_transformed_list = []
        for i in range(len(dataset)):
            image2D = np.reshape(dataset.iloc[:,1:].values[i], original_shape)
            image2D = torch.tensor(image2D, dtype=torch.float32)
            image2D_transformed = transform(image2D)
            image2D_transformed
            image1D_transformed = torch.reshape(image2D_transformed, (-1,))
            image1D_transformed_list.append(image1D_transformed)
        # drop original images and add transformed images
        dataset.drop(dataset.columns[1:], axis=1, inplace=True)
        # Convert a list to a numpy array
        transform_images_array = np.array([tensor.numpy() for tensor in image1D_transformed_list])
        # Create a DataFrame from the numpy array
        transform_images_df = pd.DataFrame(transform_images_array)
        # Concatenate the DataFrames
        dataset = pd.concat([dataset, transform_images_df], axis=1)
        print("new image size: ", dataset.iloc[:, 1:].values[0].shape)

        return dataset
    
    def train_file(self):
        vision_train = pd.read_csv(os.path.join(self.vision_dir, 'mnist_train.csv'), header=None)
        vision_train.rename(columns={0: "labels"}, inplace=True)
        
        if self.transform_flag == "True":
            vision_train = self.transform(transform_type=self.transform_type, resize_ratio=self.resize_ratio, dataset=vision_train)

        return vision_train
    
    def test_file(self):
        vision_test = pd.read_csv(os.path.join(self.vision_dir, 'mnist_test.csv'), header=None)
        vision_test.rename(columns={0: "labels"}, inplace=True)
        
        if self.transform_flag == "True":
            vision_test = self.transform(transform_type=self.transform_type, resize_ratio=self.resize_ratio, dataset=vision_test)

        return vision_test
    
    def labels(self):
        vision_train_labels = self.vision_train["labels"].tolist()
        print(f"The number of train labels: {len(vision_train_labels)}")
        vision_test_labels = self.vision_test["labels"].tolist()
        print(f"The number of test labels: {len(vision_test_labels)}")

        return vision_train_labels, vision_test_labels

    def features(self):
        print("parsing train features...")
        # normalize each row of vision_train[:, 1:]
        vision_train_features = self.vision_train.iloc[:, 1:].apply(lambda x: (x-np.mean(x))/np.std(x), axis=1)

        print("parsing test features...")
        # normalize each row of vision_test[:, 1:]
        vision_test_features = self.vision_test.iloc[:, 1:].apply(lambda x: (x-np.mean(x))/np.std(x), axis=1)

        # convert to list
        vision_train_features = vision_train_features.values.tolist()
        vision_test_features = vision_test_features.values.tolist()

        return vision_train_features, vision_test_features

# Custom Vision Dataset
class Vision_Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
