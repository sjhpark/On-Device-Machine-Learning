from utils import *
from data_processing import *
from models import *
from arguments import arguments
import torch

def main(dataset, lr, batch_size, num_hidden, epochs, hidden_dim, out_dim, bias, val):
    
    # config
    config = load_yaml('config')

    # device 
    device = torch.device(config['device'])
    print(f"Device: {device}")

    # flags
    print(f"Bias: {bias}")
    print(f"Validation mode: {val}")

     # criterion (loss function)
    criterion = nn.CrossEntropyLoss()
    print(f"Loss function: {criterion.__class__.__name__}")

    if dataset == 'SST':
        lang_train_labels, lang_dev_labels = SSTDataProcessor().labels()
        lang_train_features, lang_dev_features = SSTDataProcessor().features()
        input_dim = len(lang_train_features[0])
        model = FFNN(input_dim, hidden_dim['SST'], out_dim['SST'], num_hidden, bias).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        lang_train_dataset = Lang_Dataset(lang_train_features, lang_train_labels)
        lang_train_dataloader = DataLoader(lang_train_dataset, batch_size=batch_size, shuffle=True)

        lang_dev_dataset = Lang_Dataset(lang_dev_features, lang_dev_labels)
        lang_dev_dataloader = DataLoader(lang_dev_dataset, batch_size=batch_size, shuffle=True)

        train(model=model, 
        criterion=criterion,
        optimizer=optimizer, 
        epochs=epochs,
        train_dataloader=lang_train_dataloader, 
        val_dataloader=lang_dev_dataloader,
        test_dataset = lang_dev_dataset,
        device=device,
        val = val)

    elif dataset == 'MNIST':
        vision_train_labels, vision_test_labels = MNISTDataProcessor().labels()
        vision_train_features, vision_test_features = MNISTDataProcessor().features()
        input_dim = len(vision_train_features[0])
        model = FFNN(input_dim, hidden_dim['MNIST'], out_dim['MNIST'], num_hidden, bias).to(device)
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
        test_dataset = vision_test_dataset,
        device=device,
        val = val)

if __name__ == '__main__':
    main(**arguments())