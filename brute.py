"""Iterate over every combination of hyperparameters."""
import logging
from network import Network
from tqdm import tqdm

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='brute-log.txt'
)

def train_networks(networks):
    """Train each network.

    Args:
        networks (list): Current population of networks
    """
    pbar = tqdm(total = len(networks))
    for network in networks:
        network.train()
        network.print_network()
        pbar.update(1)
    pbar.close()
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])


def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def generate_network_list(nn_param_choices):
    """Generate a list of all possible networks.

    Args:
        nn_param_choices (dict): The parameter choices

    Returns:
        networks (list): A list of network objects
    """
    networks = []

    for nb_neurons in nn_param_choices['nb_neurons']:
        for nb_layers in nn_param_choices['nb_layers']:
            for activation in nn_param_choices['activation']:
                for optimizer in nn_param_choices['optimizer']:
                    network = {
                        'nb_neurons' :nb_neurons,
                        'activation' :activation,
                        'nb_layers'  :nb_layers,
                        'optimizer'  : optimizer
                    }
                    network_obj = Network()
                    network_obj.create_set(network)
                    networks.append(network_obj)
    return networks
def main():
    """Brute force test every network."""
    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }
    networks = generate_network_list(nn_param_choices=nn_param_choices)
    train_networks(networks)

if __name__ == '__main__':
    main()
