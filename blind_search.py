import numpy as np
import random
from typing import Tuple
from shared.solution import Solution
from shared.functions import Function


class BlindSearchAlgorithm:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def blind_search(self, func_name: str, bounds: Tuple[float, float], 
                    max_generations: int = 100, population_size: int = 30, 
                    dimension: int = 2, minimize: bool = True, verbose: bool = True) -> dict:
        
        function = Function(func_name)
        
        # Step 1: Generate initial random solution x_b
        x_b = Solution(dimension, bounds[0], bounds[1])
        x_b.generate_random()
        x_b.evaluate(function)
        
        history = [(x_b.parameters[0], x_b.parameters[1], x_b.f, True)]
        generation = 0
        total_evaluations = 1
        
        if verbose:
            print(f"Initial: f = {x_b.f:.6f}")
        
        # Step 2: Main loop - while g < g_maximal
        while generation < max_generations:
            generation += 1
            
            # Step 3: Generate NP solutions for this generation
            generation_solutions = []
            
            for i in range(population_size):
                solution = Solution(dimension, bounds[0], bounds[1])
                solution.generate_random()
                solution.evaluate(function)
                generation_solutions.append(solution)
                total_evaluations += 1
                
                # Add to history (not marked as best yet)
                history.append((solution.parameters[0], solution.parameters[1], solution.f, False))
            
            # Step 4: Select best solution x_s from this generation
            if minimize:
                x_s = min(generation_solutions, key=lambda sol: sol.f)
            else:
                x_s = max(generation_solutions, key=lambda sol: sol.f)
            
            # Mark the best solution from this generation in history
            best_idx = generation_solutions.index(x_s)
            history[-population_size + best_idx] = (x_s.parameters[0], x_s.parameters[1], x_s.f, True)
            
            # Step 5: Compare f(x_s) < f(x_b) for minimization
            is_better = (x_s.f < x_b.f) if minimize else (x_s.f > x_b.f)
            
            if is_better:
                # Step 6a: x_b = x_s (replace with better solution)
                x_b = x_s.copy()
                
                if verbose:
                    print(f"Gen {generation}: improved to {x_b.f:.6f}")
            else:
                if verbose and generation % 10 == 0:  # Print every 10 generations when no improvement
                    print(f"Gen {generation}: no improvement (best: {x_b.f:.6f})")
        
        # Step 7: Return x_b
        return {
            'algorithm': 'Blind Search',
            'best_solution': x_b,
            'best_position': x_b.parameters,
            'best_value': x_b.f,
            'history': history,
            'generations': max_generations,
            'population_size': population_size,
            'function_evaluations': total_evaluations
        }