import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.functions import (get_function_categories, get_visualization_bounds)
from hill_climbing import HillClimbingAlgorithm
from shared.visualization import Visualizer
from shared.utils import create_output_directory

def demonstrate_hill_climbing():
    print("Testing Hill Climbing algorithm:")
    test_functions = ['sphere', 'ackley', 'rastrigin', 'rosenbrock']
    
    for func_name in test_functions:
        bounds = get_visualization_bounds(func_name)
        
        if func_name in ['ackley', 'rastrigin']:
            sigma = 0.3
            num_neighbors = 10
        else:
            sigma = 0.1
            num_neighbors = 5
        
        hc = HillClimbingAlgorithm(seed=42)
        result = hc.hill_climbing(
            func_name, bounds, 
            max_generations=50, 
            num_neighbors=num_neighbors,
            sigma=sigma,
            dimension=2,
            verbose=False
        )
        
        best_sol = result['best_solution']
        print(f"  {func_name.capitalize():<12} best: {best_sol.f:.6f}  improvements: {result['improvements']}/{result['generations']}")
    
    print()


def demonstrate_hill_climbing_visualization():
    print("Creating hill climbing visualizations...")
    
    output_dir = create_output_directory("Exercise-2")
    visualizer = Visualizer(resolution=50)
    hc = HillClimbingAlgorithm(seed=42)
    
    categories = get_function_categories()
    
    for category, functions in categories.items():
        for func_name in functions:
            vis_bounds = get_visualization_bounds(func_name)
            
            if func_name in ['ackley', 'rastrigin']:
                search_bounds = (-8, 8)
                sigma = 0.3
                num_neighbors = 8
                generations = 40
            elif func_name in ['griewank', 'schwefel']:
                search_bounds = (-20, 20)
                sigma = 2.0
                num_neighbors = 10
                generations = 50
            elif func_name in ['levy', 'michalewicz']:
                search_bounds = vis_bounds
                sigma = 0.15
                num_neighbors = 8
                generations = 35
            else:
                search_bounds = (-3, 3)
                sigma = 0.1
                num_neighbors = 6
                generations = 30
            
            result = hc.hill_climbing(
                func_name, search_bounds,
                max_generations=generations, 
                num_neighbors=num_neighbors,
                sigma=sigma,
                dimension=2,
                verbose=False
            )
            
            filename = f"hill_climbing_{func_name}_3d.png"
            fig, ax = visualizer.plot_search_trajectory_3d(
                func_name, result['history'], vis_bounds, vis_bounds,
                save_path=f"{output_dir}/{filename}", algorithm_name="Hill Climbing"
            )
            plt.close(fig)
            
            best_sol = result['best_solution']
            print(f"  {func_name.capitalize():<12} visualization saved: {filename}")
    
    print("All visualizations completed.")
    print()

def main():
    print("TASK 2: Hill Climbing Algorithm")
    print()
    
    demonstrate_hill_climbing()
    demonstrate_hill_climbing_visualization()
    print("\nTask 2 completed.")


if __name__ == "__main__":
    main()
