import numpy as np
import random
from typing import Tuple
from copy import deepcopy
from shared.solution import Solution
from shared.functions import Function


class Particle:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lB = lower_bound
        self.uB = upper_bound
        self.position = np.zeros(self.dimension)
        self.velocity = np.zeros(self.dimension)
        self.f = np.inf  # current fitness value
        self.pBest_position = np.zeros(self.dimension)  # personal best position
        self.pBest_f = np.inf  # personal best fitness value
    
    def generate_random(self):
        self.position = np.random.uniform(self.lB, self.uB, self.dimension)
        # Initialize velocity to small random values
        v_max = (self.uB - self.lB) * 0.1  # 10% of search space
        self.velocity = np.random.uniform(-v_max, v_max, self.dimension)
    
    def evaluate(self, function):
        self.f = function.evaluate(self.position)
    
    def update_personal_best(self, minimize=True):
        if minimize:
            if self.f < self.pBest_f:
                self.pBest_position = self.position.copy()
                self.pBest_f = self.f
                return True
        else:
            if self.f > self.pBest_f:
                self.pBest_position = self.position.copy()
                self.pBest_f = self.f
                return True
        return False
    
    def copy(self):
        new_particle = Particle(self.dimension, self.lB, self.uB)
        new_particle.position = self.position.copy()
        new_particle.velocity = self.velocity.copy()
        new_particle.f = self.f
        new_particle.pBest_position = self.pBest_position.copy()
        new_particle.pBest_f = self.pBest_f
        return new_particle


class ParticleSwarmOptimization:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def particle_swarm_optimization(self, func_name: str, bounds: Tuple[float, float], 
                                  pop_size: int = 15, c1: float = 2.0, c2: float = 2.0,
                                  w: float = 0.9, w_min: float = 0.4, M_max: int = 50, 
                                  dimension: int = 2, minimize: bool = True, 
                                  verbose: bool = True) -> dict:
        function = Function(func_name)
        
        # Step 1: Generate pop_size random individuals (particles)
        swarm = []
        for i in range(pop_size):
            particle = Particle(dimension, bounds[0], bounds[1])
            particle.generate_random()
            particle.evaluate(function)
            # Initialize personal best
            particle.pBest_position = particle.position.copy()
            particle.pBest_f = particle.f
            swarm.append(particle)
        
        # Step 2: Select the best individual from the population (gBest)
        if minimize:
            gBest_particle = min(swarm, key=lambda p: p.f)
        else:
            gBest_particle = max(swarm, key=lambda p: p.f)
        
        gBest_position = gBest_particle.position.copy()
        gBest_f = gBest_particle.f
        
        # Initialize history - track all evaluated solutions
        # For 2D visualization: (x, y, f_value, is_best_in_generation)
        history = []
        for particle in swarm:
            is_best = (particle.f == gBest_f)
            history.append((particle.position[0], particle.position[1], particle.f, is_best))
        
        # Velocity bounds (10% of search space)
        v_max = (bounds[1] - bounds[0]) * 0.1
        v_min = -v_max
        
        m = 0
        total_evaluations = pop_size
        improvements = 0
        
        if verbose:
            print(f"Initial swarm: best f = {gBest_f:.6f}")
        
        # Step 3: Main loop
        while m < M_max:
            m += 1
            
            # Update inertia weight linearly from w to w_min
            current_w = w - (w - w_min) * (m / M_max)
            
            # Step 4: For each particle in swarm
            for i, particle in enumerate(swarm):
                
                # Step 5: Calculate new velocity for particle
                # v = w * v + c1 * r1 * (pBest - x) + c2 * r2 * (gBest - x)
                r1 = np.random.uniform(0, 1, dimension)
                r2 = np.random.uniform(0, 1, dimension)
                
                cognitive_component = c1 * r1 * (particle.pBest_position - particle.position)
                social_component = c2 * r2 * (gBest_position - particle.position)
                
                new_velocity = (current_w * particle.velocity + 
                               cognitive_component + 
                               social_component)
                
                # Check boundaries of velocity (v_min, v_max)
                new_velocity = np.clip(new_velocity, v_min, v_max)
                particle.velocity = new_velocity
                
                # Step 6: Calculate new position for particle
                # Old position is always replaced by new position
                new_position = particle.position + particle.velocity
                
                # CHECK BOUNDARIES!
                new_position = np.clip(new_position, bounds[0], bounds[1])
                particle.position = new_position
                
                # Step 7: Evaluate new position
                particle.evaluate(function)
                total_evaluations += 1
                
                # Step 8: Compare new position to its pBest
                if particle.update_personal_best(minimize):
                    improvements += 1
                    
                    # Step 9: If pBest is better than gBest, update gBest
                    if minimize:
                        if particle.pBest_f < gBest_f:
                            gBest_position = particle.pBest_position.copy()
                            gBest_f = particle.pBest_f
                    else:
                        if particle.pBest_f > gBest_f:
                            gBest_position = particle.pBest_position.copy()
                            gBest_f = particle.pBest_f
                
                # Add current position to history
                history.append((particle.position[0], particle.position[1], particle.f, False))
            
            # Mark the best solution from this generation in history
            # Find current generation's best particle
            if minimize:
                gen_best_particle = min(swarm, key=lambda p: p.f)
            else:
                gen_best_particle = max(swarm, key=lambda p: p.f)
            
            # Mark the best particle from this generation in history
            gen_start_idx = len(history) - pop_size
            for idx in range(pop_size):
                if swarm[idx].f == gen_best_particle.f:
                    hist_idx = gen_start_idx + idx
                    if hist_idx < len(history):
                        x_val, y_val, f_val, _ = history[hist_idx]
                        history[hist_idx] = (x_val, y_val, f_val, True)
                    break
            
            if verbose and (m % 10 == 0 or gBest_f == gen_best_particle.f):
                if gBest_f == gen_best_particle.f and m > 1:
                    print(f"Iteration {m}: improved to {gBest_f:.6f} (w={current_w:.3f})")
                elif m % 10 == 0:
                    print(f"Iteration {m}: best = {gBest_f:.6f} (w={current_w:.3f})")
        
        if verbose:
            print(f"Final: best f = {gBest_f:.6f}")
            print(f"Total improvements: {improvements}")
        
        # Create best solution object for compatibility
        best_solution = Solution(dimension, bounds[0], bounds[1])
        best_solution.parameters = gBest_position
        best_solution.f = gBest_f
        
        return {
            'algorithm': 'Particle Swarm Optimization',
            'best_solution': best_solution,
            'best_position': gBest_position,
            'best_value': gBest_f,
            'history': history,
            'iterations': M_max,
            'population_size': pop_size,
            'c1': c1,
            'c2': c2,
            'w_initial': w,
            'w_final': w_min,
            'function_evaluations': total_evaluations,
            'improvements': improvements
        }
