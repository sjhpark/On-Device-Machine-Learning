from utils import *
from data_processing import *
from models import *
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset type (MNIST or SST)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_hidden', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--hidden_dim', type=int, default={"MNIST": 1024, "SST": 256}, help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default={"MNIST": 10, "SST": 2} , help='output dimension')
    args = parser.parse_args()

    # config
    config = load_yaml('config')

    # criterion (loss function)
    criterion = nn.CrossEntropyLoss()

    # device 
    device = torch.device(config['device'])

    if args.dataset == 'SST':
        lang_train_labels, lang_dev_labels = SSTDataProcessor().labels()
        lang_train_features, lang_dev_features = SSTDataProcessor().features()
        input_dim = len(lang_train_features[0])
        model = FFNN(input_dim, args.hidden_dim['SST'], args.out_dim['SST'], args.num_hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        lang_train_dataset = Lang_Dataset(lang_train_features, lang_train_labels)
        lang_train_dataloader = DataLoader(lang_train_dataset, batch_size=args.batch_size, shuffle=True)

        lang_dev_dataset = Lang_Dataset(lang_dev_features, lang_dev_labels)
        lang_dev_dataloader = DataLoader(lang_dev_dataset, batch_size=args.batch_size, shuffle=True)

        train(model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        epochs=args.epochs,
        train_dataloader=lang_train_dataloader, 
        dev_dataloader=lang_dev_dataloader, 
        device=device)


    elif args.dataset == 'MNIST':
        vision_train_labels, vision_test_labels = MNISTDataProcessor().labels()
        vision_train_features, vision_test_features = MNISTDataProcessor().features()
        input_dim = len(vision_train_features[0])
        model = FFNN(input_dim, args.hidden_dim['MNIST'], args.out_dim['MNIST'], args.num_hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        vision_train_dataset = Vision_Dataset(vision_train_features, vision_train_labels)
        vision_train_dataloader = DataLoader(vision_train_dataset, batch_size=args.batch_size, shuffle=True)

        vision_test_dataset = Vision_Dataset(vision_test_features, vision_test_labels)
        vision_test_dataloader = DataLoader(vision_test_dataset, batch_size=args.batch_size, shuffle=True)

        train(model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        epochs=args.epochs,
        train_dataloader=vision_train_dataloader, 
        dev_dataloader=vision_test_dataloader, 
        device=device)

    

