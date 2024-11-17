import logging
from optimizer import Optimizer
from network import Network
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)
def train_networks(networks):
    pbar = tqdm(total = len(networks))
    for network in networks:
        network.train()
        network.print_network()
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    total_accuracy = 0
    for network in networks:
         total_accuracy = total_accuracy + network.accuracy
    return total_accuracy/len(networks)

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def generate(population,generations,nn_param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks

    """
    optimizer = Optimizer(nn_param_choices)
    pop = optimizer.create_population(population)
    #evolve
    for generation in range(1,generations+1):
        logging.info("***Doing generation %d of %d***" %(generation, generations))
        train_networks(pop)
        average_accuracy = get_average_accuracy(pop)
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        if generation != generations:
            pop = optimizer.evolve(pop)


def main():
    population = 20
    generations = 10
    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }
   
    logging.info("***Evolving %d generations with population %d***" %(generations, population))
    #funtion generate 
    generate(population,generations,nn_param_choices)

if __name__ == '__main__':
    main()