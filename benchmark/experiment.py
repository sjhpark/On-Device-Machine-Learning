from train import main
from arguments import arguments

class Experiment:
    def __init__(self):
        self.param_dict = arguments()

    def depth_analysis(self):
        # Experiment: Nerual Network Depth Variation
        for i in [2, 3, 4, 5]:
            print(f"\n-------Experiment: Neural Network Depth Variation -- Number of Hidden Layers: {i}-------")
            self.param_dict['num_hidden'] = i
            main(**self.param_dict)

    def width_analysis(self):
        # Experiment: Neural Network Width Variation
        for i in [1024//2, 1024, 1024*2, 1024*4]:
            print(f"\n-------Experiment: Neural Network Width Variation -- Hidden Layer Width: {i}-------")
            self.param_dict['hidden_dim']['MNIST'] = i
            main(**self.param_dict)

    def detph_width_analysis(self):
        # Experiment: Neural Network Depth and Width Variation
        for i, j in zip([2, 3, 4, 5], [1024//2, 1024, 1024*2, 1024*4]):
                print(f"\n-------Experiment: Neural Network Depth and Width Variation -- Number of Hidden Layers: {i} & Hidden Layer Width: {j}-------")
                self.param_dict['num_hidden'] = i
                self.param_dict['hidden_dim']['MNIST'] = j
                main(**self.param_dict)

if __name__ == '__main__':
    # Experiment().depth_analysis()
    # Experiment().width_analysis()
    Experiment().detph_width_analysis()