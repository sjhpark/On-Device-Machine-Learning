import argparse

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset type (MNIST or SST)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_hidden', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--hidden_dim', type=int, default={"MNIST": 1024, "SST": 256}, help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default={"MNIST": 10, "SST": 2} , help='output dimension')
    parser.add_argument('--PTQ_type', type=str, default='None', help='Post-Training Quantization type (dynamic or static)')
    parser.add_argument('--q_domain', type=str, default='None', help='target quantization domain of your model (e.g. qint8, float16, etc.)')
    parser.add_argument('--bias', action='store_true', help='bias') # True if called, else False
    parser.add_argument('--val', action='store_true', help='validate the model') # True if called, else False
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
