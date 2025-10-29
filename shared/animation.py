import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from .functions import Function, get_function_optimum


class AnimationCreator:
    def __init__(self, resolution=100):
        self.resolution = resolution
    
    def create_meshgrid(self, func_name, x_range, y_range):
        x = np.linspace(x_range[0], x_range[1], self.resolution)
        y = np.linspace(y_range[0], y_range[1], self.resolution)
        X, Y = np.meshgrid(x, y)
        
        function = Function(func_name)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function.evaluate(np.array([X[i, j], Y[i, j]]))
        
        return X, Y, Z
    
    def create_algorithm_animation_2d(self, func_name, history, bounds, 
                                     save_path=None, algorithm_name="Algorithm",
                                     fps=10, population_size=20):
        # Create meshgrid for the function
        X, Y, Z = self.create_meshgrid(func_name, bounds, bounds)
        
        # Calculate number of generations
        num_generations = len(history) // population_size
        if num_generations == 0:
            num_generations = 1
        
        # Generate frames
        frames = []
        
        for gen in range(num_generations):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot heatmap
            heatmap = ax.contourf(X, Y, Z, levels=50, cmap='jet', alpha=0.9)
            ax.contour(X, Y, Z, levels=15, colors='black', alpha=0.2, linewidths=0.5)
            
            # Get points up to current generation
            end_idx = min((gen + 1) * population_size, len(history))
            current_history = history[:end_idx]
            
            if current_history:
                # All evaluated points
                all_x = [point[0] for point in current_history]
                all_y = [point[1] for point in current_history]
                
                # Best points trajectory
                best_points = [point for point in current_history if point[3]]
                
                # Show all points (only recent generation)
                if gen > 0:
                    start_idx = gen * population_size
                    recent_x = all_x[start_idx:end_idx]
                    recent_y = all_y[start_idx:end_idx]
                    ax.scatter(recent_x, recent_y, c='gray', s=15, alpha=0.5, marker='o', zorder=5)
                
                # Show best trajectory
                if len(best_points) >= 2:
                    best_x = [point[0] for point in best_points]
                    best_y = [point[1] for point in best_points]
                    
                    ax.plot(best_x, best_y, color='white', linewidth=3, alpha=0.9, zorder=10)
                    ax.plot(best_x, best_y, color='lime', linewidth=2, alpha=1.0, zorder=11)
                    
                    # Mark intermediate best points
                    if len(best_points) > 2:
                        ax.scatter(best_x[1:-1], best_y[1:-1], c='yellow', s=40, 
                                  edgecolors='black', linewidth=1, alpha=0.9, 
                                  marker='o', zorder=12)
                    
                    # Mark start
                    ax.scatter([best_x[0]], [best_y[0]], c='green', s=150, 
                              marker='o', edgecolors='black', linewidth=2,
                              label='Start', zorder=15)
                    
                    # Mark current best
                    ax.scatter([best_x[-1]], [best_y[-1]], c='red', s=150, 
                              marker='s', edgecolors='black', linewidth=2,
                              label='Current best', zorder=15)
                    
                    # Show current best value
                    current_best_f = best_points[-1][2]
                    ax.text(0.02, 0.98, f'Generation: {gen}\nBest: {current_best_f:.6f}', 
                           transform=ax.transAxes, fontsize=14, fontweight='bold',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95))
            
            # Mark global optimum
            optimum_point, optimum_value = get_function_optimum(func_name)
            if optimum_point is not None:
                if (bounds[0] <= optimum_point[0] <= bounds[1] and 
                    bounds[0] <= optimum_point[1] <= bounds[1]):
                    ax.scatter([optimum_point[0]], [optimum_point[1]], 
                              c='white', s=200, marker='*', 
                              edgecolors='black', linewidth=2,
                              label='Global optimum', zorder=20)
            
            # Add colorbar to all frames for consistent size
            cbar = plt.colorbar(heatmap, ax=ax, pad=0.02)
            cbar.ax.set_ylabel('Function Value', fontsize=12, rotation=270, labelpad=20)
            
            ax.set_xlabel('$x_1$', fontsize=14)
            ax.set_ylabel('$x_2$', fontsize=14)
            ax.set_title(f'{func_name.capitalize()} Function - {algorithm_name}', 
                        fontsize=16, fontweight='bold', pad=15)
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[0], bounds[1])
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            plt.tight_layout()
            
            # Save frame to memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            
            # Read as PIL Image
            img = Image.open(buf)
            frames.append(img.copy())
            buf.close()
            plt.close(fig)
        
        # Save as GIF
        if save_path and frames:
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,  # duration in milliseconds
                loop=0,
                optimize=False
            )
            print(f"    Created GIF with {len(frames)} frames")
        
        return len(frames)

