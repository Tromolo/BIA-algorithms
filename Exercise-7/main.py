import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.functions import (get_function_categories, get_function_domain, get_effective_visualization_bounds)
from soma import SOMAAlgorithm
from shared.visualization import Visualizer
from shared.animation import AnimationCreator
from shared.utils import create_output_directory

def demonstrate_soma():
    """Demonstrate SOMA on basic test functions"""
    print("=" * 60)
    print("SOMA Basic Demonstration")
    print("=" * 60)
    
    test_functions = ['sphere', 'ackley', 'rastrigin', 'rosenbrock']
    
    pop_size = 20
    PRT = 0.4
    PathLength = 3.0
    Step = 0.11
    M_max = 100
    
    for func_name in test_functions:
        bounds = get_function_domain(func_name)
        
        soma = SOMAAlgorithm(seed=42)
        result = soma.soma(
            func_name, bounds,
            pop_size=pop_size,
            PRT=PRT,
            PathLength=PathLength,
            Step=Step,
            M_max=M_max,
            dimension=2,
            verbose=False
        )
        
        best_sol = result['best_solution']
        print(f"  {func_name.capitalize():<12} best: {best_sol.f:.6f}  improvements: {result['improvements']}/{result['function_evaluations']}")
    
    print()


def demonstrate_soma_visualization():
    """Create 3D visualizations for SOMA"""
    print("=" * 60)
    print("SOMA 3D Visualization")
    print("=" * 60)
    
    output_dir = create_output_directory("Exercise-7")
    visualizer = Visualizer(resolution=50)
    
    pop_size = 20
    PRT = 0.4
    PathLength = 3.0
    Step = 0.11
    M_max = 100
    
    soma = SOMAAlgorithm(seed=42)
    
    categories = get_function_categories()
    
    for category, functions in categories.items():
        for func_name in functions:
            if func_name == 'griewank':
                search_bounds = (-20, 20)  
            else:
                search_bounds = get_function_domain(func_name)
            
            vis_bounds = get_effective_visualization_bounds(func_name)
            
            result = soma.soma(
                func_name, search_bounds,
                pop_size=pop_size,
                PRT=PRT,
                PathLength=PathLength,
                Step=Step,
                M_max=M_max,
                dimension=2,
                verbose=False
            )
            
            filename = f"soma_{func_name}_3d.png"
            fig, ax = visualizer.plot_search_trajectory_3d(
                func_name, result['history'], vis_bounds, vis_bounds,
                save_path=f"{output_dir}/{filename}", 
                algorithm_name="SOMA"
            )
            plt.close(fig)
            
            best_sol = result['best_solution']
            print(f"  {func_name.capitalize():<12} visualization saved: {filename} (best: {best_sol.f:.6f})")
    
    print("All visualizations completed.")
    print()


def demonstrate_soma_parameters():
    """Test different SOMA parameter configurations"""
    print("=" * 60)
    print("SOMA Parameter Analysis")
    print("=" * 60)
    
    param_configs = [
        {"PRT": 0.4, "PathLength": 3.0, "Step": 0.11, "name": "Standard SOMA (PRT=0.4, PathLength=3.0, Step=0.11)"},
        {"PRT": 0.2, "PathLength": 3.0, "Step": 0.11, "name": "Low perturbation (PRT=0.2)"},
        {"PRT": 0.6, "PathLength": 3.0, "Step": 0.11, "name": "High perturbation (PRT=0.6)"},
        {"PRT": 0.4, "PathLength": 2.0, "Step": 0.11, "name": "Short path (PathLength=2.0)"},
        {"PRT": 0.4, "PathLength": 3.0, "Step": 0.05, "name": "Fine steps (Step=0.05)"}
    ]
    
    pop_size = 20
    M_max = 100
    bounds = get_function_domain('rastrigin')
    
    for config in param_configs:
        soma = SOMAAlgorithm(seed=42)
        result = soma.soma(
            'rastrigin', bounds,
            pop_size=pop_size,
            PRT=config["PRT"],
            PathLength=config["PathLength"],
            Step=config["Step"],
            M_max=M_max,
            dimension=2,
            verbose=False
        )
        
        print(f"  {config['name']:<50} Best: {result['best_solution'].f:.6f}, Improvements: {result['improvements']}")
    
    print()


def create_comprehensive_comparison():
    """Run SOMA on all functions and create comprehensive analysis"""
    print("=" * 60)
    print("SOMA Comprehensive Analysis")
    print("=" * 60)
    
    output_dir = create_output_directory("Exercise-7")
    
    pop_size = 20
    PRT = 0.4
    PathLength = 3.0
    Step = 0.11
    M_max = 100
    
    results_summary = []
    categories = get_function_categories()
    
    for category, functions in categories.items():
        print(f"\n  {category}:")
        for func_name in functions:
            bounds = get_function_domain(func_name)
            if func_name == 'griewank':
                bounds = (-20, 20)
            
            soma = SOMAAlgorithm(seed=42)
            result = soma.soma(
                func_name, bounds,
                pop_size=pop_size,
                PRT=PRT,
                PathLength=PathLength,
                Step=Step,
                M_max=M_max,
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
    """Create a comprehensive grid visualization of all SOMA results"""
    print("=" * 60)
    print("SOMA Comprehensive Grid Visualization")
    print("=" * 60)
    
    output_dir = create_output_directory("Exercise-7")
    visualizer = Visualizer(resolution=50)
    
    pop_size = 20
    PRT = 0.4
    PathLength = 3.0
    Step = 0.11
    M_max = 100
    
    results_dict = {}
    categories = get_function_categories()
    
    for category, functions in categories.items():
        for func_name in functions:
            if func_name == 'griewank':
                search_bounds = (-20, 20)
            else:
                search_bounds = get_function_domain(func_name)
            
            soma = SOMAAlgorithm(seed=42)
            result = soma.soma(
                func_name, search_bounds,
                pop_size=pop_size,
                PRT=PRT,
                PathLength=PathLength,
                Step=Step,
                M_max=M_max,
                dimension=2,
                verbose=False
            )
            
            results_dict[func_name] = result
            print(f"  {func_name.capitalize():<12} completed (best: {result['best_solution'].f:.4f})")
    
    # Use the same grid visualization method as other algorithms
    fig = visualizer.plot_all_differential_evolution_grid(
        results_dict, 
        save_path=f"{output_dir}/soma_comprehensive_grid.png",
        algorithm_name="SOMA"
    )
    plt.close(fig)
    
    print("Comprehensive grid visualization completed.")


def create_heatmap_visualizations():
    """Create heatmap visualizations for selected functions"""
    print("=" * 60)
    print("SOMA Heatmap Visualizations")
    print("=" * 60)
    
    output_dir = create_output_directory("Exercise-7")
    visualizer = Visualizer(resolution=100)
    
    pop_size = 20
    PRT = 0.4
    PathLength = 3.0
    Step = 0.11
    M_max = 100

    selected_functions = {
        'sphere': (-20, 20),
        'ackley': (-20, 20),
        'rastrigin': (-5.12, 5.12)
    }
    
    for func_name, vis_bounds in selected_functions.items():
        soma = SOMAAlgorithm(seed=42)
        
        result = soma.soma(
            func_name, vis_bounds,
            pop_size=pop_size,
            PRT=PRT,
            PathLength=PathLength,
            Step=Step,
            M_max=M_max,
            dimension=2,
            verbose=False
        )
        
        filename = f"soma_{func_name}_heatmap.png"
        fig, ax = visualizer.plot_search_trajectory_2d_heatmap(
            func_name, result['history'], 
            vis_bounds, vis_bounds,
            save_path=f"{output_dir}/{filename}", 
            algorithm_name="SOMA",
            show_all_points=True
        )
        plt.close(fig)
        
        best_sol = result['best_solution']
        print(f"  {func_name.capitalize():<12} heatmap saved: {filename} (best: {best_sol.f:.6f})")
    
    print("Heatmap visualizations completed.")


def create_animated_gifs():
    """Create animated GIFs showing SOMA evolution"""
    print("=" * 60)
    print("SOMA Animated GIF Creation")
    print("=" * 60)
    
    output_dir = create_output_directory("Exercise-7")
    gifs_dir = f"{output_dir}/gifs"
    os.makedirs(gifs_dir, exist_ok=True)
    
    animator = AnimationCreator(resolution=80)
    soma = SOMAAlgorithm(seed=42)
    
    pop_size = 20
    PRT = 0.4
    PathLength = 3.0
    Step = 0.11
    M_max = 100
    
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
        result = soma.soma(
            func_name, bounds,
            pop_size=pop_size,
            PRT=PRT,
            PathLength=PathLength,
            Step=Step,
            M_max=M_max,
            dimension=2,
            verbose=False
        )
        
        gif_path = f"{gifs_dir}/soma_{func_name}_evolution.gif"
        num_frames = animator.create_algorithm_animation_2d(
            func_name, 
            result['history'], 
            bounds,
            save_path=gif_path,
            algorithm_name="SOMA",
            fps=5,  # 5 frames per second
            population_size=pop_size
        )
        
        print(f"  {func_name.capitalize():<12} GIF created: soma_{func_name}_evolution.gif ({num_frames} frames)")
    
    print("Animated GIF visualizations completed.")


def main():
    """Main function to run all SOMA demonstrations"""
    print("Starting SOMA (Self-organizing Migrating Algorithm) Exercise...")
    print("Parameters: pop_size=20, PRT=0.4, PathLength=3.0, Step=0.11, M_max=100")
    print()
    
    demonstrate_soma()
    demonstrate_soma_parameters()
    demonstrate_soma_visualization()
    create_heatmap_visualizations()
    create_comprehensive_comparison()
    create_comprehensive_grid_visualization()
    create_animated_gifs()
    
    print("\n" + "=" * 60)
    print("Exercise 7 - SOMA completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
