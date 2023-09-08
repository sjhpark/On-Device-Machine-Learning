from train import main
from arguments import arguments

class Experiment:
    def __init__(self):
        self.param_dict = arguments()

    def depth_analysis(self):
        # Experiment: Nerual Network Depth Variation
        for i in [1, 2, 3, 4, 5]:
            print(f"\n-------Experiment: Neural Network Depth Variation -- Number of Hidden Layers: {i}-------")
            self.param_dict['num_hidden'] = i
            main(**self.param_dict)

    def width_analysis(self):
        # Experiment: Neural Network Width Variation
        for i in [1024//4, 1024//2, 1024, 1024*2, 1024*4]:
            print(f"\n-------Experiment: Neural Network Width Variation -- Hidden Layer Width: {i}-------")
            self.param_dict['hidden_dim']['MNIST'] = i
            main(**self.param_dict)

    def input_size_analysis(self):
        # Experiment: Input Size Variation
        return

if __name__ == '__main__':
    # Experiment().depth_analysis()
    Experiment().width_analysis()