
import random
from network import Network
class Optimizer():
    def __init__(self,nn_param_choices, retain=0.4, random_select=0.1, mutate_chance=0.2):
        self.nn_param_choice = nn_param_choices
        self.accuracy = 0
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.visited_solutions = []
    def create_population(self, count):
        population = []
        
        for _ in range(count):
            network = {}
            for key in self.nn_param_choice:
                network[key] = random.choice(self.nn_param_choice[key])
            self.visited_solutions.append(network)
            network_obj = Network()
            network_obj.create_set(network)
            population.append(network_obj)
        
        return population
    
    def fitness(self,network):
        return network.accuracy
    
    def mutate(self, network):
        while(network.network not in self.visited_solutions):
            genType = random.choice(list(self.nn_param_choice.keys()))
            network.network[genType] = random.choice(self.nn_param_choice[genType])
        self.visited_solutions.append(network.network)
        return network

    def breed(self, Dad, Mom):
        babies = []

        for _ in range(2):
            child = {}
            
            for param in self.nn_param_choice:
                child[param] = random.choice([Dad.network[param],Mom.network[param]])
            network = Network()
            network.create_set(child)
            if self.mutate_chance > random.random():
                network = self.mutate(network)
            if network.network in self.visited_solutions:
                self.mutate(network)
            self.visited_solutions.append(network.network)
            babies.append(network)
            
        return babies
    
    def evolve(self,pop):
        graded = [(self.fitness(network),network) for network in pop]

        graded = [x[1] for x in sorted(graded, key = lambda x:x[0],reverse= True)]
        retain_length = int(len(graded)*self.retain)
        parents = graded[:retain_length]

        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
        
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        while len(children) < desired_length:
            Dad = random.randint(0,parents_length-1)
            Mom = random.randint(0,parents_length-1)
            if Dad != Mom:
                Dad = parents[Dad]
                Mom = parents[Mom]

                babies = self.breed(Dad,Mom)
                for baby in babies :
                    if len(children) < desired_length :
                        children.append(baby)
        parents.extend(children)
        return parents