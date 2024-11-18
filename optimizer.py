import random
from network import Network

class Optimizer():
    def __init__(self, nn_param_choices, retain=0.4, random_select=0.1, mutate_chance=0.2):
        self.nn_param_choice = nn_param_choices
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.visited_solutions = set()  # Sử dụng set để lưu trữ các giải pháp đã thử nghiệm

    def create_population(self, count):
        """Tạo quần thể ban đầu với các cá thể không trùng lặp."""
        population = []
        
        while len(population) < count:
            network = {}
            for key in self.nn_param_choice:
                network[key] = random.choice(self.nn_param_choice[key])
            network_tuple = tuple(network.items())  # Mã hóa thành tuple để lưu trong set
            
            if network_tuple not in self.visited_solutions:
                self.visited_solutions.add(network_tuple)
                network_obj = Network()
                network_obj.create_set(network)
                population.append(network_obj)
        
        return population

    def fitness(self, network):
        """Tính độ thích nghi dựa trên độ chính xác."""
        return network.accuracy

    def mutate(self, network):
        """Đột biến mạng nơ-ron, đảm bảo không trùng lặp."""
        max_attempts = 100  # Giới hạn số lần thử đột biến
        attempts = 0
        
        while attempts < max_attempts:
            genType = random.choice(list(self.nn_param_choice.keys()))
            original_value = network.network[genType]
            network.network[genType] = random.choice(self.nn_param_choice[genType])
            
            network_tuple = tuple(network.network.items())
            if network_tuple not in self.visited_solutions:
                self.visited_solutions.add(network_tuple)
                return network
            
            # Hoàn nguyên nếu trùng lặp
            network.network[genType] = original_value
            attempts += 1
        
        return network  

    def breed(self, Dad, Mom):
        """Lai ghép hai mạng nơ-ron, đảm bảo con không trùng lặp."""
        babies = []

        for _ in range(2):
            child = {}
            
            for param in self.nn_param_choice:
                child[param] = random.choice([Dad.network[param], Mom.network[param]])
            
            network = Network()
            network.create_set(child)
            network_tuple = tuple(network.network.items())
            
            if self.mutate_chance > random.random():
                network = self.mutate(network)
            
            if network_tuple not in self.visited_solutions:
                self.visited_solutions.add(network_tuple)
                babies.append(network)
            else:
                network = self.mutate(network)
                self.visited_solutions.add(tuple(network.network.items()))
                babies.append(network)
            
        return babies

    def evolve(self, pop):
        """Tiến hóa quần thể qua các thế hệ."""
        # Tính độ thích nghi và sắp xếp quần thể
        graded = [(self.fitness(network), network) for network in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        
        # Chọn các cá thể tốt nhất
        retain_length = int(len(graded) * self.retain)
        parents = graded[:retain_length]

        # Chọn ngẫu nhiên một số cá thể khác để duy trì tính đa dạng
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
        
        # Lai ghép để tạo thế hệ con
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        while len(children) < desired_length:
            Dad_idx = random.randint(0, parents_length - 1)
            Mom_idx = random.randint(0, parents_length - 1)
            if Dad_idx != Mom_idx:
                Dad = parents[Dad_idx]
                Mom = parents[Mom_idx]

                babies = self.breed(Dad, Mom)
                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)
        
        # Kết hợp thế hệ cha mẹ và con
        parents.extend(children)
        return parents
