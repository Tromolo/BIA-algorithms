import numpy as np
import random
from typing import Tuple
from copy import deepcopy
from shared.solution import Solution
from shared.functions import Function


class SOMAAlgorithm:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def soma(self, func_name: str, bounds: Tuple[float, float], 
             pop_size: int = 20, PRT: float = 0.4, PathLength: float = 3.0,
             Step: float = 0.11, M_max: int = 100, dimension: int = 2, 
             minimize: bool = True, verbose: bool = True) -> dict:

        function = Function(func_name)
        
        # Step 1: Generate pop_size random individuals
        population = []
        for i in range(pop_size):
            individual = Solution(dimension, bounds[0], bounds[1])
            individual.generate_random()
            individual.evaluate(function)
            population.append(individual)
        
        # Find initial leader
        if minimize:
            leader = min(population, key=lambda sol: sol.f).copy()
        else:
            leader = max(population, key=lambda sol: sol.f).copy()
        
        # Initialize history - track all evaluated solutions
        # For 2D visualization: (x, y, f_value, is_best_in_generation)
        history = []
        for sol in population:
            is_best = (sol.f == leader.f)
            history.append((sol.parameters[0], sol.parameters[1], sol.f, is_best))
        
        migration = 0
        total_evaluations = pop_size
        improvements = 0
        
        if verbose:
            print(f"Initial population: leader f = {leader.f:.6f}")
        
        # Step 2: Main migration loop
        while migration < M_max:
            migration += 1
            
            # Create new population
            new_population = deepcopy(population)
            
            # Step 3: For each individual in population
            for i, individual in enumerate(population):
                
                # Skip if this individual is the current leader
                if individual.f == leader.f:
                    continue
                
                # Step 4: Generate PRT Vector
                # PRT vector determines which dimensions will be perturbed
                prt_vector = np.random.uniform(0, 1, dimension) < PRT
                
                if not np.any(prt_vector):
                    prt_vector[np.random.randint(0, dimension)] = True
                
                # Step 5: Migration towards leader
                best_position = individual.parameters.copy()
                best_fitness = individual.f
                
                # Calculate direction vector from individual to leader
                direction = leader.parameters - individual.parameters
                
                # Step 6: Move along the path towards leader
                t = Step  # Start with first step
                while t <= PathLength:
                    
                    # Calculate new position
                    # new_pos = start_pos + t * (leader_pos - start_pos) * PRT_vector
                    new_position = individual.parameters + t * direction * prt_vector
                    
                    # Ensure new position respects bounds
                    new_position = np.clip(new_position, bounds[0], bounds[1])
                    
                    # Evaluate new position
                    trial_solution = Solution(dimension, bounds[0], bounds[1])
                    trial_solution.parameters = new_position
                    trial_solution.evaluate(function)
                    total_evaluations += 1
                    
                    # Add to history
                    history.append((new_position[0], new_position[1], trial_solution.f, False))
                    
                    # Check if this position is better than current best for this individual
                    is_better = (trial_solution.f < best_fitness) if minimize else (trial_solution.f > best_fitness)
                    
                    if is_better:
                        best_position = new_position.copy()
                        best_fitness = trial_solution.f
                        improvements += 1
                    
                    # Move to next step
                    t += Step
                
                # Step 7: Update individual with best position found during migration
                new_population[i].parameters = best_position
                new_population[i].f = best_fitness
            
            # Step 8: Replace old population with new population
            population = new_population
            
            # Step 9: Find new leader
            if minimize:
                new_leader = min(population, key=lambda sol: sol.f)
            else:
                new_leader = max(population, key=lambda sol: sol.f)
            
            # Update global leader if improved
            leader_improved = False
            if minimize and new_leader.f < leader.f:
                leader = new_leader.copy()
                leader_improved = True
                if verbose:
                    print(f"Migration {migration}: improved to {leader.f:.6f}")
            elif not minimize and new_leader.f > leader.f:
                leader = new_leader.copy()
                leader_improved = True
                if verbose:
                    print(f"Migration {migration}: improved to {leader.f:.6f}")
            elif verbose and migration % 10 == 0:
                print(f"Migration {migration}: leader = {leader.f:.6f}")
            
            # Mark the leader from this migration in history
            # Simply mark the most recent entry that matches the leader
            for idx in range(len(population)):
                if population[idx].f == new_leader.f:
                    # Mark the most recent matching entry as best
                    for hist_idx in range(len(history) - 1, max(0, len(history) - 100), -1):
                        x_val, y_val, f_val, _ = history[hist_idx]
                        if abs(x_val - population[idx].parameters[0]) < 1e-10 and abs(y_val - population[idx].parameters[1]) < 1e-10:
                            history[hist_idx] = (x_val, y_val, f_val, True)
                            break
                    break
        
        if verbose:
            print(f"Final: leader f = {leader.f:.6f}")
            print(f"Total improvements: {improvements}")
        
        return {
            'algorithm': 'SOMA',
            'best_solution': leader,
            'best_position': leader.parameters,
            'best_value': leader.f,
            'history': history,
            'migrations': M_max,
            'population_size': pop_size,
            'PRT': PRT,
            'PathLength': PathLength,
            'Step': Step,
            'function_evaluations': total_evaluations,
            'improvements': improvements
        }
