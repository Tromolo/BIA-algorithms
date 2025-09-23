import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.solution import Solution
from shared.functions import (Function, get_function_domain, 
                       get_all_functions, get_function_categories, get_visualization_bounds)
from blind_search import BlindSearchAlgorithm
from shared.visualization import Visualizer
from shared.utils import create_output_directory


def demonstrate_functions():
    print("Testing optimization functions:")
    test_params = np.array([1.0, 1.0])
    
    for func_name in get_all_functions():
        func = Function(func_name)
        value = func.evaluate(test_params)
        domain = get_function_domain(func_name)
        
        domain_str = f"[{domain[0]:.1f}, {domain[1]:.1f}]²"
        
        print(f"  {func_name.capitalize():<12} f({test_params}) = {value:<12.6f} domain: {domain_str}")
    print()


def demonstrate_3d_visualizations():
    print("Creating 3D function visualizations...")
    
    output_dir = create_output_directory("Exercise-1")
    visualizer = Visualizer(resolution=60)
    
    fig = visualizer.plot_all_functions_grid(save_path=f"{output_dir}/all_functions_3d.png")
    plt.close(fig)
    print(f"  Saved: all_functions_3d.png")
    print()


def demonstrate_blind_search():
    print("Testing Blind Search algorithm:")
    test_functions = ['sphere', 'ackley', 'rastrigin', 'rosenbrock']
    
    for func_name in test_functions:
        bounds = get_visualization_bounds(func_name)
        search = BlindSearchAlgorithm(seed=42)
        result = search.blind_search(
            func_name, bounds, 
            max_generations=30, 
            population_size=25, 
            dimension=2,
            verbose=False
        )
        
        best_sol = result['best_solution']
        print(f"  {func_name.capitalize():<12} best: {best_sol.f:.6f}  evaluations: {result['function_evaluations']}")
    
    print()


def demonstrate_search_visualization():
    print("Creating blind search visualizations...")
    
    output_dir = create_output_directory("Exercise-1")
    visualizer = Visualizer(resolution=50)
    search = BlindSearchAlgorithm(seed=42)
    
    categories = get_function_categories()
    
    for category, functions in categories.items():
        for func_name in functions:
            vis_bounds = get_visualization_bounds(func_name)
            
            result = search.blind_search(
                func_name, vis_bounds, 
                max_generations=40, 
                population_size=20, 
                dimension=2,
                verbose=False
            )
            
            filename = f"blind_search_{func_name}_3d.png"
            fig, ax = visualizer.plot_search_trajectory_3d(
                func_name, result['history'], vis_bounds, vis_bounds,
                save_path=f"{output_dir}/{filename}", algorithm_name="Blind Search"
            )
            plt.close(fig)
            
            best_sol = result['best_solution']
            print(f"  {func_name.capitalize():<12} visualization saved: {filename}")
    
    print("All visualizations completed.")
    print()

def demonstrate_solution_class():
    print("Testing Solution class:")
    
    sol = Solution(dimension=2, lower_bound=-5, upper_bound=5)
    sol.generate_random()
    
    sphere_func = Function('sphere')
    sol.evaluate(sphere_func)
    
    print(f"  Random solution: f = {sol.f:.6f}, params = [{sol.parameters[0]:.4f}, {sol.parameters[1]:.4f}]")
    
    sol2 = sol.copy()
    print(f"  Copied solution: f = {sol2.f:.6f}, params = [{sol2.parameters[0]:.4f}, {sol2.parameters[1]:.4f}]")
    print()

def main():
    print("TASK 1: Optimization Functions and Blind Search")
    print()
    
    demonstrate_solution_class()
    demonstrate_functions()
    demonstrate_3d_visualizations()
    demonstrate_blind_search()
    demonstrate_search_visualization()
    print("\nTask 1 completed.")


if __name__ == "__main__":
    main()