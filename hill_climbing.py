import numpy as np
import random
from typing import Tuple
from shared.solution import Solution
from shared.functions import Function


class HillClimbingAlgorithm:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def hill_climbing(self, func_name: str, bounds: Tuple[float, float], 
                     max_generations: int = 100, num_neighbors: int = 5,
                     sigma: float = 0.1, dimension: int = 2, 
                     minimize: bool = True, verbose: bool = True) -> dict:

        function = Function(func_name)
        
        # Step 1: Generate initial random solution x_b
        x_b = Solution(dimension, bounds[0], bounds[1])
        x_b.generate_random()
        x_b.evaluate(function)
        
        history = [(x_b.parameters[0], x_b.parameters[1], x_b.f, True)]  # (x, y, f_value, is_current)
        generation = 0
        total_evaluations = 1
        improvements = 0
        
        if verbose:
            print(f"Initial: f = {x_b.f:.6f}")
        
        # Step 2: Main loop - while g < g_maximal
        while generation < max_generations:
            generation += 1
            
            # Step 3: Generate neighbors using normal distribution N(x_b, σ)
            neighbors = []
            for i in range(num_neighbors):
                neighbor = Solution(dimension, bounds[0], bounds[1])
                
                # Generate neighbor using normal distribution around current solution
                neighbor_params = np.random.normal(x_b.parameters, sigma)
                
                # Ensure bounds are respected
                neighbor_params = np.clip(neighbor_params, bounds[0], bounds[1])
                neighbor.parameters = neighbor_params
                
                # Evaluate neighbor
                neighbor.evaluate(function)
                neighbors.append(neighbor)
                total_evaluations += 1
                
                # Add to history (not marked as current yet)
                history.append((neighbor.parameters[0], neighbor.parameters[1], neighbor.f, False))
            
            # Step 4: Evaluate x_s (find best neighbor)
            if minimize:
                best_neighbor = min(neighbors, key=lambda sol: sol.f)
            else:
                best_neighbor = max(neighbors, key=lambda sol: sol.f)
            
            # Step 5: Compare f(x_s) < f(x_b) for minimization
            is_better = (best_neighbor.f < x_b.f) if minimize else (best_neighbor.f > x_b.f)
            
            if is_better:
                # Step 6a: x_b = x_s (replace with better neighbor)
                x_b = best_neighbor.copy()
                improvements += 1
                
                # Mark this as the new current solution in the trajectory
                best_idx = neighbors.index(best_neighbor)
                history[-num_neighbors + best_idx] = (best_neighbor.parameters[0], 
                                                    best_neighbor.parameters[1], 
                                                    best_neighbor.f, True)
                
                if verbose:
                    print(f"Gen {generation}: improved to {x_b.f:.6f}")
            else:
                if verbose and generation % 10 == 0:  # Print every 10 generations when no improvement
                    print(f"Gen {generation}: no improvement (best: {x_b.f:.6f})")
        
        # Calculate improvement statistics
        improvement_rate = (improvements / max_generations) * 100
        
        return {
            'algorithm': 'Hill Climbing',
            'best_solution': x_b,
            'best_position': x_b.parameters,
            'best_value': x_b.f,
            'history': history,
            'generations': max_generations,
            'num_neighbors': num_neighbors,
            'sigma': sigma,
            'function_evaluations': total_evaluations,
            'improvements': improvements,
            'improvement_rate': improvement_rate
        }