import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.functions import (get_function_categories, get_function_domain, get_effective_visualization_bounds)
from simulated_annealing import SimulatedAnnealingAlgorithm
from shared.visualization import Visualizer
from shared.utils import create_output_directory

def demonstrate_simulated_annealing():
    test_functions = ['sphere', 'ackley', 'rastrigin', 'rosenbrock']
    
    for func_name in test_functions:
        bounds = get_function_domain(func_name)
        
        if func_name in ['ackley', 'rastrigin']:
            T_0 = 200
            alpha = 0.95
            sigma = 0.5
            max_iterations = 800
        elif func_name == 'rosenbrock':
            T_0 = 150
            alpha = 0.97
            sigma = 0.3
            max_iterations = 1000
        else:  # sphere
            T_0 = 100
            alpha = 0.95
            sigma = 0.2
            max_iterations = 600
        
        sa = SimulatedAnnealingAlgorithm(seed=42)
        result = sa.simulated_annealing(
            func_name, bounds, 
            T_0=T_0,
            T_min=0.1,
            alpha=alpha,
            max_iterations=max_iterations,
            sigma=sigma,
            dimension=2,
            verbose=False
        )
        
        best_sol = result['best_solution']
        print(f"  {func_name.capitalize():<12} best: {best_sol.f:.6f}  accepted: {result['accepted_moves']}/{result['iterations']} ({result['acceptance_rate']:.1f}%)")
    
    print()


def demonstrate_simulated_annealing_visualization():    
    output_dir = create_output_directory("Exercise-3")
    visualizer = Visualizer(resolution=50)
    sa = SimulatedAnnealingAlgorithm(seed=42)
    
    categories = get_function_categories()
    
    for category, functions in categories.items():
        for func_name in functions:
            if func_name == 'griewank':
                search_bounds = (-20, 20)  
            else:
                search_bounds = get_function_domain(func_name)
            vis_bounds = get_effective_visualization_bounds(func_name)
            
            if func_name in ['ackley', 'rastrigin']:
                T_0 = 200
                alpha = 0.95
                sigma = 0.4
                max_iterations = 600
            elif func_name in ['griewank', 'schwefel']:
                T_0 = 50
                alpha = 0.95
                sigma = 0.3
                max_iterations = 800
            elif func_name in ['levy', 'michalewicz']:
                T_0 = 150
                alpha = 0.97
                sigma = 0.25
                max_iterations = 700
            elif func_name == 'rosenbrock':
                T_0 = 180
                alpha = 0.97
                sigma = 0.3
                max_iterations = 1000
            else:  # sphere, zakharov
                T_0 = 120
                alpha = 0.95
                sigma = 0.2
                max_iterations = 600
            
            result = sa.simulated_annealing(
                func_name, search_bounds,
                T_0=T_0,
                T_min=0.05,
                alpha=alpha,
                max_iterations=max_iterations,
                sigma=sigma,
                dimension=2,
                verbose=False
            )
            
            filename = f"simulated_annealing_{func_name}_3d.png"
            fig, ax = visualizer.plot_search_trajectory_3d(
                func_name, result['history'], vis_bounds, vis_bounds,
                save_path=f"{output_dir}/{filename}", algorithm_name="Simulated Annealing"
            )
            plt.close(fig)
            
            best_sol = result['best_solution']
            print(f"  {func_name.capitalize():<12} visualization saved: {filename} (best: {best_sol.f:.6f}, acceptance: {result['acceptance_rate']:.1f}%)")
    
    print("All visualizations completed.")
    print()


def demonstrate_temperature_cooling():
    print("Demonstrating temperature cooling process:")
    
    cooling_configs = [
        {"T_0": 100, "alpha": 0.90, "name": "Fast cooling (α=0.90)"},
        {"T_0": 100, "alpha": 0.95, "name": "Medium cooling (α=0.95)"},
        {"T_0": 100, "alpha": 0.99, "name": "Slow cooling (α=0.99)"}
    ]
    
    sa = SimulatedAnnealingAlgorithm(seed=42)
    
    for config in cooling_configs:
        result = sa.simulated_annealing(
            'ackley', get_function_domain('ackley'),
            T_0=config["T_0"],
            T_min=0.1,
            alpha=config["alpha"],
            max_iterations=200,
            sigma=0.3,
            dimension=2,
            verbose=False
        )
        
        print(f"  {config['name']:<25} Final T: {result['T_final']:.4f}, Best: {result['best_solution'].f:.6f}, Acceptance: {result['acceptance_rate']:.1f}%")
    
    print()


def create_comprehensive_grid_visualization():    
    output_dir = create_output_directory("Exercise-3")
    visualizer = Visualizer(resolution=50)
    sa = SimulatedAnnealingAlgorithm(seed=42)
    
    results_dict = {}
    categories = get_function_categories()
    
    for category, functions in categories.items():
        for func_name in functions:
            if func_name == 'griewank':
                search_bounds = (-20, 20)
            else:
                search_bounds = get_function_domain(func_name)
            
            if func_name in ['ackley', 'rastrigin']:
                T_0 = 200
                alpha = 0.95
                sigma = 0.4
                max_iterations = 1000
            elif func_name in ['griewank', 'schwefel']:
                T_0 = 50
                alpha = 0.95
                sigma = 0.3
                max_iterations = 1000
            elif func_name in ['levy', 'michalewicz']:
                T_0 = 150
                alpha = 0.97
                sigma = 0.25
                max_iterations = 1000
            elif func_name == 'rosenbrock':
                T_0 = 180
                alpha = 0.97
                sigma = 0.3
                max_iterations = 1000
            else:  # sphere, zakharov
                T_0 = 120
                alpha = 0.95
                sigma = 0.2
                max_iterations = 1000
            
            result = sa.simulated_annealing(
                func_name, search_bounds,
                T_0=T_0,
                T_min=0.05,
                alpha=alpha,
                max_iterations=max_iterations,
                sigma=sigma,
                dimension=2,
                verbose=False
            )
            
            results_dict[func_name] = result
            print(f"  {func_name.capitalize():<12} completed (best: {result['best_solution'].f:.4f})")
    
    fig = visualizer.plot_all_simulated_annealing_grid(
        results_dict, 
        save_path=f"{output_dir}/simulated_annealing_comprehensive_grid.png"
    )
    plt.close(fig)
    

def main():    
    demonstrate_simulated_annealing()
    demonstrate_temperature_cooling()
    demonstrate_simulated_annealing_visualization()
    create_comprehensive_grid_visualization()


if __name__ == "__main__":
    main()
