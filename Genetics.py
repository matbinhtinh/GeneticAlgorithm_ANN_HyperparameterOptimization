from optimizer import Optimizer
from network import Network
from tqdm import tqdm

def train_networks(networks):
    """Train all networks in the given list."""
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train()
        network.print_network()
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Calculate the average accuracy of the given list of networks."""
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy
    return total_accuracy / len(networks)

def print_networks(networks):
    """Print a list of networks."""
    for network in networks:
        network.print_network()

def generate(population, generations, nn_param_choices):
    """Generate a network with the genetic algorithm."""
    optimizer = Optimizer(nn_param_choices)
    pop = optimizer.create_population(population)
    
    # Evolve
    for generation in range(1, generations + 1):
        print("***Doing generation %d of %d***" % (generation, generations))  # For console output
        train_networks(networks=pop)
        average_accuracy = get_average_accuracy(pop)
        print("Generation average: %.2f%%" % (average_accuracy * 100))  # For console output
        print('-' * 80)

        if generation != generations:
            pop = optimizer.evolve(pop)

    # Sort our final population by accuracy.
    pop = sorted(pop, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(pop[:5])

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
   
    print("***Evolving %d generations with population %d***" % (generations, population))  # For console output
    generate(population, generations, nn_param_choices)

if __name__ == '__main__':
    main()
