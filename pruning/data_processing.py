import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from indiv_utils import load_yaml

# configs
config = load_yaml('config')

####################  MNIST Vision Data ####################
# MNIST Data Processor
class MNISTDataProcessor:
    def __init__(self):
        self.vision_dir = config['paths']['MNIST_data_dir']

        self.transform_flag = "True"
        self.transform_type = config['processing']['transform']
        self.resize_ratio = 1 # the number to divide the original image size by

        self.vision_train = self.train_file()
        self.vision_test = self.test_file()

    def transform(self, transform_type, resize_ratio, dataset):
        # Resize images (by Resize or CenterCrop)
        image_size = dataset.iloc[:, 1:].values[0].shape
        original_shape = (1, int(image_size[0]**0.5), int(image_size[0]**0.5))
        new_image_size = 20 # 20 pixels
        
        if transform_type == 'Resize':
            print(f"Resizing test images from {int(image_size[0]**0.5)}x{int(image_size[0]**0.5)} to {new_image_size}x{new_image_size}")
            transform = transforms.Resize((new_image_size, new_image_size), antialias=True)
        elif transform_type == "Center Crop":
            print(f"Center Cropping images from {int(image_size[0]**0.5)}x{int(image_size[0]**0.5)} to {new_image_size}x{new_image_size}")
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
    
    def features_and_labels(self):
        """features"""
        print("parsing train features...")
        # normalize each row of vision_train[:, 1:]
        vision_train_features = self.vision_train.iloc[:, 1:].apply(lambda x: (x-np.mean(x))/np.std(x), axis=1)

        print("parsing test features...")
        # normalize each row of vision_test[:, 1:]
        vision_test_features = self.vision_test.iloc[:, 1:].apply(lambda x: (x-np.mean(x))/np.std(x), axis=1)

        # convert to list
        vision_train_features = vision_train_features.values.tolist()
        vision_test_features = vision_test_features.values.tolist()

        """labels"""
        vision_train_labels = self.vision_train["labels"].tolist()
        print(f"The number of train labels: {len(vision_train_labels)}")
        vision_test_labels = self.vision_test["labels"].tolist()
        print(f"The number of test labels: {len(vision_test_labels)}")

        return vision_train_features, vision_test_features, vision_train_labels, vision_test_labels

    def vision_test_dataset(self):
        # test features
        print("parsing test features...")
        # normalize each row of vision_test[:, 1:]
        vision_test_features = self.vision_test.iloc[:, 1:].apply(lambda x: (x-np.mean(x))/np.std(x), axis=1)
        vision_test_features = vision_test_features.values.tolist()

        # test labels
        vision_test_labels = self.vision_test["labels"].tolist()
        print(f"The number of test labels: {len(vision_test_labels)}")

        vision_test_dataset = Vision_Dataset(vision_test_features, vision_test_labels)
        return vision_test_dataset

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
