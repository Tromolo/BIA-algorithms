import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from genetic_algorithm import GeneticAlgorithm, City, Tour
from shared.tsp_visualization import TSPVisualizer


def create_output_directory(task_name: str = "Exercise-4") -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"results/{task_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_ga_and_visualize():    
    output_dir = create_output_directory("Exercise-4")
    visualizer = TSPVisualizer()
    
    for num_cities in [20, 30, 40]:
        print(f"\n  Processing {num_cities} cities...")
        
        ga = GeneticAlgorithm(seed=42)
        result = ga.genetic_algorithm(
            num_cities=num_cities,
            population_size=20,
            max_generations=200,
            mutation_rate=0.5,
            verbose=False
        )
        
        filename = f"ga_{num_cities}_cities_best_tour.png"
        fig = visualizer.plot_tour(
            result['cities'],
            result['best_tour'].route,
            title=f"TSP Solution - {num_cities} Cities (Distance: {result['best_distance']:.2f})",
            save_path=f"{output_dir}/{filename}",
            show_labels=True
        )
        plt.close(fig)
        
        print(f"    Best Distance: {result['best_distance']:.2f}")
        print(f"    Saved: {filename}")


def main():
    run_ga_and_visualize()


if __name__ == "__main__":
    main()

