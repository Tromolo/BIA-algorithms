import numpy as np
import random
from typing import Tuple
from copy import deepcopy
from shared.solution import Solution
from shared.functions import Function


class DifferentialEvolutionAlgorithm:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def differential_evolution(self, func_name: str, bounds: Tuple[float, float], 
                              NP: int = 20, F: float = 0.5, CR: float = 0.5,
                              G_maxim: int = 50, dimension: int = 2, 
                              minimize: bool = True, verbose: bool = True) -> dict:

        function = Function(func_name)
        
        # Step 1: Generate NP random individuals (initial population)
        pop = []
        for i in range(NP):
            individual = Solution(dimension, bounds[0], bounds[1])
            individual.generate_random()
            individual.evaluate(function)
            pop.append(individual)
        
        # Find initial best solution
        if minimize:
            best_solution = min(pop, key=lambda sol: sol.f).copy()
        else:
            best_solution = max(pop, key=lambda sol: sol.f).copy()
        
        # Initialize history - track all evaluated solutions
        # For 2D visualization: (x, y, f_value, is_best_in_generation)
        history = []
        for sol in pop:
            is_best = (sol.f == best_solution.f)
            history.append((sol.parameters[0], sol.parameters[1], sol.f, is_best))
        
        g = 0
        total_evaluations = NP
        improvements = 0
        
        if verbose:
            print(f"Initial population: best f = {best_solution.f:.6f}")
        
        # Step 2: Main loop
        while g < G_maxim:
            g += 1
            
            # Create new generation (deep copy of current population)
            new_pop = deepcopy(pop)
            
            # Step 3: For each individual in population
            for i, x in enumerate(pop):  # x is the target vector
                
                # Step 4: Select three random distinct indices r1, r2, r3
                # such that r1 != r2 != r3 != i
                available_indices = [idx for idx in range(NP) if idx != i]
                r1, r2, r3 = random.sample(available_indices, 3)
                
                # Step 5: Mutation - create mutation vector v
                # v = x_r3 + F * (x_r1 - x_r2)
                v = pop[r3].parameters + F * (pop[r1].parameters - pop[r2].parameters)
                
                # Ensure mutation vector respects bounds
                v = np.clip(v, bounds[0], bounds[1])
                
                # Step 6: Crossover - create trial vector u
                u = np.zeros(dimension)
                j_rnd = np.random.randint(0, dimension)  # Ensure at least one parameter from v
                
                for j in range(dimension):
                    # Binomial crossover
                    if np.random.uniform() < CR or j == j_rnd:
                        u[j] = v[j]  # Take from mutation vector
                    else:
                        u[j] = x.parameters[j]  # Take from target vector
                
                # Step 7: Evaluate trial vector
                trial_solution = Solution(dimension, bounds[0], bounds[1])
                trial_solution.parameters = u
                trial_solution.evaluate(function)
                f_u = trial_solution.f
                total_evaluations += 1
                
                # Step 8: Selection - compare trial with target
                # We accept if trial is better OR EQUAL
                is_better_or_equal = (f_u <= x.f) if minimize else (f_u >= x.f)
                
                if is_better_or_equal:
                    # Replace target with trial in new population
                    new_pop[i] = trial_solution
                    
                    # Track if this is an improvement (strictly better)
                    if (minimize and f_u < x.f) or (not minimize and f_u > x.f):
                        improvements += 1
                
                # Add trial to history (will mark best later)
                history.append((trial_solution.parameters[0], trial_solution.parameters[1], f_u, False))
            
            # Step 9: Replace old population with new population
            pop = new_pop
            
            # Find best in current generation
            if minimize:
                gen_best = min(pop, key=lambda sol: sol.f)
            else:
                gen_best = max(pop, key=lambda sol: sol.f)
            
            # Update global best if improved
            if minimize and gen_best.f < best_solution.f:
                best_solution = gen_best.copy()
                if verbose:
                    print(f"Gen {g}: improved to {best_solution.f:.6f}")
            elif not minimize and gen_best.f > best_solution.f:
                best_solution = gen_best.copy()
                if verbose:
                    print(f"Gen {g}: improved to {best_solution.f:.6f}")
            elif verbose and g % 10 == 0:
                print(f"Gen {g}: best = {best_solution.f:.6f}")
            
            # Mark the best solution from this generation in history
            # The last NP entries in history are from this generation
            gen_start_idx = len(history) - NP
            for idx in range(NP):
                if pop[idx].f == gen_best.f:
                    hist_idx = gen_start_idx + idx
                    if hist_idx < len(history):
                        x_val, y_val, f_val, _ = history[hist_idx]
                        history[hist_idx] = (x_val, y_val, f_val, True)
                    break
        
        if verbose:
            print(f"Final: best f = {best_solution.f:.6f}")
            print(f"Total improvements: {improvements}")
        
        return {
            'algorithm': 'Differential Evolution',
            'best_solution': best_solution,
            'best_position': best_solution.parameters,
            'best_value': best_solution.f,
            'history': history,
            'generations': G_maxim,
            'population_size': NP,
            'F': F,
            'CR': CR,
            'function_evaluations': total_evaluations,
            'improvements': improvements
        }

