import numpy as np
from typing import List, Tuple, Optional


class City:
    def __init__(self, x: float, y: float, name: Optional[str] = None):
        self.x = x
        self.y = y
        self.name = name
    
    def distance_to(self, city: 'City') -> float:
        return np.sqrt((self.x - city.x)**2 + (self.y - city.y)**2) # Euclidean distance
    
    def __repr__(self):
        if self.name:
            return f"City({self.name}: {self.x:.1f}, {self.y:.1f})"
        return f"City({self.x:.1f}, {self.y:.1f})"


class Tour:
    def __init__(self, cities: List[City], route: Optional[List[int]] = None):
        self.cities = cities
        self.num_cities = len(cities)
        
        if route is None:
            self.route = list(np.random.permutation(self.num_cities))
        else:
            self.route = route.copy()
        
        self.distance = None
        self.fitness = None
    
    def calculate_distance(self) -> float:
        total_distance = 0.0
        for i in range(self.num_cities):
            from_city = self.cities[self.route[i]]
            to_city = self.cities[self.route[(i + 1) % self.num_cities]]  # back to start by %
            total_distance += from_city.distance_to(to_city)
        
        self.distance = total_distance
        self.fitness = 1.0 / (1.0 + total_distance)  # fitness is inverse of distance
        return total_distance
    
    def copy(self) -> 'Tour':
        new_tour = Tour(self.cities, self.route)
        new_tour.distance = self.distance
        new_tour.fitness = self.fitness
        return new_tour
    
    def __repr__(self):
        return f"Tour(distance={self.distance:.2f}, route={self.route[:5]}...)"


class GeneticAlgorithm:
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
    
    def generate_cities(self, num_cities: int, x_range: Tuple[float, float] = (0, 200),
                       y_range: Tuple[float, float] = (0, 200)) -> List[City]:
        cities = []
        for i in range(num_cities):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            cities.append(City(x, y, name=str(i)))
        return cities
    
    def crossover_order(self, parent_a: Tour, parent_b: Tour) -> Tour:
        route_a = parent_a.route
        route_b = parent_b.route
        size = len(route_a)
        
        # random crossover points
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        
        # Initialize offspring with -1 (empty positions)
        offspring_route = [-1] * size
        # copy segment from parent_A
        offspring_route[start:end] = route_a[start:end]
        
        # fill remaining positions with cities from parent_B in order
        current_pos = end
        for city in route_b[end:] + route_b[:end]:
            if city not in offspring_route:
                offspring_route[current_pos % size] = city
                current_pos += 1
        
        offspring = Tour(parent_a.cities, offspring_route)
        return offspring
    
    def mutate_swap(self, tour: Tour) -> Tour:
        mutated = tour.copy()
        # select two random positions to swap
        idx1, idx2 = np.random.choice(range(tour.num_cities), 2, replace=False)
        # swap the cities at these positions
        mutated.route[idx1], mutated.route[idx2] = mutated.route[idx2], mutated.route[idx1]
        return mutated
    
    def genetic_algorithm(self, num_cities: int = 20, population_size: int = 20,
                         max_generations: int = 200, mutation_rate: float = 0.5,
                         x_range: Tuple[float, float] = (0, 200),
                         y_range: Tuple[float, float] = (0, 200),
                         cities: Optional[List[City]] = None,
                         verbose: bool = True) -> dict:
        
        # generate cities if not provided
        if cities is None:
            cities = self.generate_cities(num_cities, x_range, y_range)
        else:
            num_cities = len(cities)
        
        # population = Generate NP random individuals
        population = [Tour(cities) for _ in range(population_size)]
        
        # Evaluate individuals within population
        for tour in population:
            tour.calculate_distance()
        
        # Track best solution for history
        best_tour = min(population, key=lambda t: t.distance).copy()
        
        # Initialize history tracking
        history = []
        history.append({
            'generation': 0,
            'best_distance': best_tour.distance,
            'route': best_tour.route.copy(),
            'avg_distance': np.mean([t.distance for t in population])
        })
        
        if verbose:
            print(f"Generation 0: Best distance = {best_tour.distance:.2f}")
        
        # for i in range(G):
        for generation in range(1, max_generations + 1):
            # new_population = copy(population)
            new_population = population.copy()
            
            # for j in range(NP):
            for j in range(population_size):
                parent_a = population[j]
                
                # parent_B = random individual from population (parent_B != parent_A)
                parent_b_idx = np.random.choice([i for i in range(population_size) if i != j])
                parent_b = population[parent_b_idx]
                
                # offspring_AB = crossover(parent_A, parent_B)
                offspring = self.crossover_order(parent_a, parent_b)
                
                # if np.random.uniform() < 0.5: offspring_AB = mutate(offspring_AB)
                if np.random.uniform() < mutation_rate:
                    offspring = self.mutate_swap(offspring)
                
                # Evaluate offspring_AB
                offspring.calculate_distance()
                
                # If f(offspring_AB) < f(parent_A): new_population[j] = offspring_AB
                if offspring.distance < parent_a.distance:
                    new_population[j] = offspring
            
            # population = new_population
            population = new_population
            
            current_best = min(population, key=lambda t: t.distance)
            if current_best.distance < best_tour.distance:
                best_tour = current_best.copy()
            
            history.append({
                'generation': generation,
                'best_distance': best_tour.distance,
                'route': best_tour.route.copy(),
                'avg_distance': np.mean([t.distance for t in population])
            })
            
            if verbose and (generation % 20 == 0 or generation == max_generations):
                avg_dist = np.mean([t.distance for t in population])
                print(f"Generation {generation}: Best = {best_tour.distance:.2f}, Avg = {avg_dist:.2f}")
        
        return {
            'best_tour': best_tour,
            'best_distance': best_tour.distance,
            'cities': cities,
            'history': history,
            'population': population,
            'generations': max_generations
        }

