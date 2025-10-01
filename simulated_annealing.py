import numpy as np
import random
from typing import Tuple
from shared.solution import Solution
from shared.functions import Function


class SimulatedAnnealingAlgorithm:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def simulated_annealing(self, func_name: str, bounds: Tuple[float, float], 
                           T_0: float = 100, T_min: float = 0.5, alpha: float = 0.95,
                           max_iterations: int = 1000, sigma: float = 0.1, 
                           dimension: int = 2, minimize: bool = True, verbose: bool = True) -> dict:

        function = Function(func_name)
        
        T = T_0
        
        # Generation of initial solution
        x = Solution(dimension, bounds[0], bounds[1])
        x.generate_random()
        
        x.evaluate(function)
        
        # Track the best solution found so far
        best_solution = x.copy()
        
        history = [(x.parameters[0], x.parameters[1], x.f, True)]  # (x, y, f_value, is_current)
        iteration = 0
        total_evaluations = 1
        accepted_moves = 0
        
        if verbose:
            print(f"Initial: f = {x.f:.6f}, T = {T:.2f}")
        
        # Main loop
        while T > T_min:
            iteration += 1
            
            # Safety check to prevent infinite loops
            if iteration > max_iterations:
                if verbose:
                    print(f"Reached maximum iterations ({max_iterations})")
                break
            
            # x_1 = generate neighbour of x in normal distribution
            x_1 = Solution(dimension, bounds[0], bounds[1])
            
            # Generate neighbor using normal distribution around current solution
            neighbor_params = np.random.normal(x.parameters, sigma)
            
            # Ensure bounds are respected
            neighbor_params = np.clip(neighbor_params, bounds[0], bounds[1])
            x_1.parameters = neighbor_params
            
            # Evaluate x_1
            x_1.evaluate(function)
            total_evaluations += 1
            
            # Add neighbor to history (not marked as current yet)
            history.append((x_1.parameters[0], x_1.parameters[1], x_1.f, False))
            
            # if f(x_1) < f(x): (assuming minimization)
            is_better = (x_1.f < x.f) if minimize else (x_1.f > x.f)
            
            if is_better:
                # x = x_1
                x = x_1.copy()
                accepted_moves += 1
                
                # Mark this as the new current solution in history
                history[-1] = (x_1.parameters[0], x_1.parameters[1], x_1.f, True)
                
                # Update best solution if this is the best found so far
                if minimize and x.f < best_solution.f:
                    best_solution = x.copy()
                elif not minimize and x.f > best_solution.f:
                    best_solution = x.copy()
                
                if verbose and iteration % 50 == 0:
                    print(f"Iter {iteration}: accepted better solution f = {x.f:.6f}, T = {T:.4f}")
            else:
                # r = random number in uniform distribution
                r = random.random()
                
                # Calculate delta: Δ = f(x_1) - f(x)
                delta = x_1.f - x.f
                
                # if r < e^(-(f(x_1)-f(x))/T): which is e^(-Δ/T)
                probability = np.exp(-delta / T)
                
                if r < probability:
                    # x = x_1
                    x = x_1.copy()
                    accepted_moves += 1
                    
                    # Mark this as the new current solution in history
                    history[-1] = (x_1.parameters[0], x_1.parameters[1], x_1.f, True)
                    
                    if verbose and iteration % 100 == 0:
                        print(f"Iter {iteration}: accepted worse solution f = {x.f:.6f} (p = {probability:.4f}), T = {T:.4f}")
                # If not accepted, the neighbor is already in history but not marked as current
            
            # T = T*alpha
            T = T * alpha
            
            # Update best solution tracking (keep track of the absolute best)
            if minimize and x.f < best_solution.f:
                best_solution = x.copy()
            elif not minimize and x.f > best_solution.f:
                best_solution = x.copy()
        
        # Calculate acceptance rate
        acceptance_rate = (accepted_moves / iteration) * 100 if iteration > 0 else 0
        
        if verbose:
            print(f"Final: best f = {best_solution.f:.6f}, final T = {T:.4f}")
            print(f"Accepted moves: {accepted_moves}/{iteration} ({acceptance_rate:.1f}%)")
        
        return {
            'algorithm': 'Simulated Annealing',
            'best_solution': best_solution,
            'current_solution': x,
            'best_position': best_solution.parameters,
            'best_value': best_solution.f,
            'history': history,
            'iterations': iteration,
            'T_0': T_0,
            'T_min': T_min,
            'T_final': T,
            'alpha': alpha,
            'sigma': sigma,
            'function_evaluations': total_evaluations,
            'accepted_moves': accepted_moves,
            'acceptance_rate': acceptance_rate
        }
