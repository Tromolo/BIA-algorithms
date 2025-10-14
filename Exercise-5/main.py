import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.functions import (get_function_categories, get_function_domain, get_effective_visualization_bounds)
from differential_evolution import DifferentialEvolutionAlgorithm
from shared.visualization import Visualizer
from shared.animation import AnimationCreator
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


def create_comprehensive_grid_visualization():
    output_dir = create_output_directory("Exercise-5")
    visualizer = Visualizer(resolution=50)
    
    NP = 20
    F = 0.5
    CR = 0.5
    G_maxim = 50
    
    results_dict = {}
    categories = get_function_categories()
    
    for category, functions in categories.items():
        for func_name in functions:
            if func_name == 'griewank':
                search_bounds = (-20, 20)
            else:
                search_bounds = get_function_domain(func_name)
            
            de = DifferentialEvolutionAlgorithm(seed=42)
            result = de.differential_evolution(
                func_name, search_bounds,
                NP=NP,
                F=F,
                CR=CR,
                G_maxim=G_maxim,
                dimension=2,
                verbose=False
            )
            
            results_dict[func_name] = result
            print(f"  {func_name.capitalize():<12} completed (best: {result['best_solution'].f:.4f})")
    
    fig = visualizer.plot_all_differential_evolution_grid(
        results_dict, 
        save_path=f"{output_dir}/differential_evolution_comprehensive_grid.png"
    )
    plt.close(fig)
    
    print("Comprehensive grid visualization completed.")


def create_heatmap_visualizations():
    output_dir = create_output_directory("Exercise-5")
    visualizer = Visualizer(resolution=100)
    
    NP = 20
    F = 0.5
    CR = 0.5
    G_maxim = 50

    selected_functions = {
        'sphere': (-20, 20),
        'ackley': (-20, 20),
        'rastrigin': (-5.12, 5.12)
    }
    
    for func_name, vis_bounds in selected_functions.items():
        de = DifferentialEvolutionAlgorithm(seed=42)
        
        result = de.differential_evolution(
            func_name, vis_bounds,
            NP=NP,
            F=F,
            CR=CR,
            G_maxim=G_maxim,
            dimension=2,
            verbose=False
        )
        
        filename = f"de_{func_name}_heatmap.png"
        fig, ax = visualizer.plot_search_trajectory_2d_heatmap(
            func_name, result['history'], 
            vis_bounds, vis_bounds,
            save_path=f"{output_dir}/{filename}", 
            algorithm_name="Differential Evolution",
            show_all_points=True
        )
        plt.close(fig)
        
        best_sol = result['best_solution']
        print(f"  {func_name.capitalize():<12} heatmap saved: {filename} (best: {best_sol.f:.6f})")
    
    print("Heatmap visualizations completed.")


def create_animated_gifs():
    output_dir = create_output_directory("Exercise-5")
    gifs_dir = f"{output_dir}/gifs"
    os.makedirs(gifs_dir, exist_ok=True)
    
    animator = AnimationCreator(resolution=80)
    de = DifferentialEvolutionAlgorithm(seed=42)
    
    NP = 20
    F = 0.5
    CR = 0.5
    G_maxim = 50
    
    animation_functions = {
        'sphere': (-5.12, 5.12),
        'ackley': (-20, 20),
        'rastrigin': (-5.12, 5.12),
        'rosenbrock': (-2.048, 2.048),
        'griewank': (-20, 20),
        'schwefel': (-400, 400),
        'levy': (-10, 10),
        'michalewicz': (0, np.pi),
        'zakharov': (-10, 10)
    }
    
    print(f"  Saving GIFs to: {gifs_dir}/")
    
    for func_name, bounds in animation_functions.items():
        result = de.differential_evolution(
            func_name, bounds,
            NP=NP,
            F=F,
            CR=CR,
            G_maxim=G_maxim,
            dimension=2,
            verbose=False
        )
        
        gif_path = f"{gifs_dir}/de_{func_name}_evolution.gif"
        num_frames = animator.create_algorithm_animation_2d(
            func_name, 
            result['history'], 
            bounds,
            save_path=gif_path,
            algorithm_name="Differential Evolution",
            fps=5,  # 5 frames per second
            population_size=NP
        )
        
        print(f"  {func_name.capitalize():<12} GIF created: de_{func_name}_evolution.gif ({num_frames} frames)")
    
    print("Animated GIF visualizations completed.")


def main():
    demonstrate_differential_evolution()
    demonstrate_de_parameters()
    demonstrate_de_visualization()
    create_heatmap_visualizations()
    create_comprehensive_comparison()
    create_comprehensive_grid_visualization()
    create_animated_gifs()
    
    print("\n" + "=" * 60)
    print("Exercise 5 completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

