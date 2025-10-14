import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.functions import (get_function_categories, get_function_domain, get_effective_visualization_bounds)
from differential_evolution import DifferentialEvolutionAlgorithm
from shared.visualization import Visualizer
from shared.utils import create_output_directory

def demonstrate_differential_evolution():
    test_functions = ['sphere', 'ackley', 'rastrigin', 'rosenbrock']
    
    NP = 20
    F = 0.5
    CR = 0.5
    G_maxim = 50
    
    for func_name in test_functions:
        bounds = get_function_domain(func_name)
        
        de = DifferentialEvolutionAlgorithm(seed=42)
        result = de.differential_evolution(
            func_name, bounds,
            NP=NP,
            F=F,
            CR=CR,
            G_maxim=G_maxim,
            dimension=2,
            verbose=False
        )
        
        best_sol = result['best_solution']
        print(f"  {func_name.capitalize():<12} best: {best_sol.f:.6f}  improvements: {result['improvements']}/{result['function_evaluations']}")
    
    print()


def demonstrate_de_visualization():
    output_dir = create_output_directory("Exercise-5")
    visualizer = Visualizer(resolution=50)
    
    NP = 20
    F = 0.5
    CR = 0.5
    G_maxim = 50
    
    de = DifferentialEvolutionAlgorithm(seed=42)
    
    categories = get_function_categories()
    
    for category, functions in categories.items():
        for func_name in functions:
            if func_name == 'griewank':
                search_bounds = (-20, 20)  
            else:
                search_bounds = get_function_domain(func_name)
            
            vis_bounds = get_effective_visualization_bounds(func_name)
            
            result = de.differential_evolution(
                func_name, search_bounds,
                NP=NP,
                F=F,
                CR=CR,
                G_maxim=G_maxim,
                dimension=2,
                verbose=False
            )
            
            filename = f"differential_evolution_{func_name}_3d.png"
            fig, ax = visualizer.plot_search_trajectory_3d(
                func_name, result['history'], vis_bounds, vis_bounds,
                save_path=f"{output_dir}/{filename}", 
                algorithm_name="Differential Evolution"
            )
            plt.close(fig)
            
            best_sol = result['best_solution']
            print(f"  {func_name.capitalize():<12} visualization saved: {filename} (best: {best_sol.f:.6f})")
    
    print("All visualizations completed.")
    print()


def demonstrate_de_parameters():
    param_configs = [
        {"F": 0.3, "CR": 0.5, "name": "Low mutation (F=0.3)"},
        {"F": 0.5, "CR": 0.5, "name": "Medium mutation (F=0.5)"},
        {"F": 0.8, "CR": 0.5, "name": "High mutation (F=0.8)"},
        {"F": 0.5, "CR": 0.3, "name": "Low crossover (CR=0.3)"},
        {"F": 0.5, "CR": 0.9, "name": "High crossover (CR=0.9)"}
    ]
    
    NP = 20
    G_maxim = 50
    bounds = get_function_domain('rastrigin')
    
    for config in param_configs:
        de = DifferentialEvolutionAlgorithm(seed=42)
        result = de.differential_evolution(
            'rastrigin', bounds,
            NP=NP,
            F=config["F"],
            CR=config["CR"],
            G_maxim=G_maxim,
            dimension=2,
            verbose=False
        )
        
        print(f"  {config['name']:<30} Best: {result['best_solution'].f:.6f}, Improvements: {result['improvements']}")
    
    print()


def create_comprehensive_comparison():
    output_dir = create_output_directory("Exercise-5")
    
    NP = 20
    F = 0.5
    CR = 0.5
    G_maxim = 50
    
    results_summary = []
    categories = get_function_categories()
    
    for category, functions in categories.items():
        print(f"\n  {category}:")
        for func_name in functions:
            bounds = get_function_domain(func_name)
            if func_name == 'griewank':
                bounds = (-20, 20)
            
            de = DifferentialEvolutionAlgorithm(seed=42)
            result = de.differential_evolution(
                func_name, bounds,
                NP=NP,
                F=F,
                CR=CR,
                G_maxim=G_maxim,
                dimension=2,
                verbose=False
            )
            
            best_sol = result['best_solution']
            results_summary.append({
                'function': func_name,
                'category': category,
                'best_value': best_sol.f,
                'evaluations': result['function_evaluations'],
                'improvements': result['improvements']
            })
            
            print(f"    {func_name.capitalize():<12} best: {best_sol.f:.6f}, evaluations: {result['function_evaluations']}, improvements: {result['improvements']}")
    
    print("\nComprehensive analysis completed.")


def main():
    print("=" * 60)
    print("EXERCISE 5: Differential Evolution Algorithm")
    print("=" * 60)
    print()
    print("Parameters:")
    print("  Population size (NP): 20")
    print("  Mutation factor (F): 0.5")
    print("  Crossover rate (CR): 0.5")
    print("  Max generations (G_maxim): 50")
    print()
    
    demonstrate_differential_evolution()
    demonstrate_de_parameters()
    demonstrate_de_visualization()
    create_comprehensive_comparison()
    
    print("\n" + "=" * 60)
    print("Exercise 5 completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

