import yaml

# yaml loader
def load_yaml(filename):
    with open(f'config/{filename}.yaml','r') as f:
        output = yaml.safe_load(f)
    return output

    