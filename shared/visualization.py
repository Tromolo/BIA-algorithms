import numpy as np
import matplotlib.pyplot as plt
from .functions import Function, get_function_domain, get_function_optimum, get_effective_visualization_bounds


class Visualizer:    
    def __init__(self, resolution=100):
        self.resolution = resolution
    
    def create_meshgrid(self, func_name, x_range=None, y_range=None):
        if x_range is None or y_range is None:
            domain = get_function_domain(func_name)
            x_range = y_range = domain
        
        x = np.linspace(x_range[0], x_range[1], self.resolution)
        y = np.linspace(y_range[0], y_range[1], self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function on meshgrid
        function = Function(func_name)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function.evaluate(np.array([X[i, j], Y[i, j]]))
        
        return X, Y, Z
    
    def plot_function_3d(self, func_name, x_range=None, y_range=None, 
                        show_optimum=True, save_path=None):
        
        X, Y, Z = self.create_meshgrid(func_name, x_range, y_range)
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        surface = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.9,
                                linewidth=0, antialiased=True, 
                                rcount=100, ccount=100, shade=True)
        
        contours = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), 
                            cmap='plasma', alpha=0.7, levels=15)
        
        # Mark global optimum if known and requested
        if show_optimum:
            optimum_point, optimum_value = get_function_optimum(func_name)
            if optimum_point is not None:
                # Check if optimum is within the plotted range
                if (x_range[0] <= optimum_point[0] <= x_range[1] and 
                    y_range[0] <= optimum_point[1] <= y_range[1]):
                    ax.scatter([optimum_point[0]], [optimum_point[1]], [optimum_value],
                             color='red', s=150, marker='*', edgecolors='black', linewidth=1)
        
        ax.set_xlabel('X₁', fontsize=14, labelpad=10)
        ax.set_ylabel('X₂', fontsize=14, labelpad=10)
        ax.set_zlabel(f'{func_name.capitalize()} Function Value', fontsize=14, labelpad=10)
        ax.set_title(f'{func_name.capitalize()} Function - 3D Surface', fontsize=16, pad=20, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=30, pad=0.1)
        cbar.ax.tick_params(labelsize=12)
        
        # Optimize viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig, ax
    
    def plot_search_trajectory_3d(self, func_name, search_history, 
                                 x_range=None, y_range=None, save_path=None, 
                                 algorithm_name="Search"):
        
        # Create base function plot
        fig, ax = self.plot_function_3d(func_name, x_range, y_range, 
                                       show_optimum=True, save_path=None)
        
        if not search_history:
            return fig, ax
        
        # Extract trajectory data (only best solutions)
        best_points = [point for point in search_history if point[3]]  # is_best = True
        
        if len(best_points) < 2:
            # If no trajectory available, show all points
            x_traj = [point[0] for point in search_history]
            y_traj = [point[1] for point in search_history]
            z_traj = [point[2] for point in search_history]
            
            # Just show the points without trajectory line
            ax.scatter(x_traj, y_traj, z_traj, 
                      color='red', s=50, alpha=0.7, label='Search Points', zorder=8)
            
            # Mark start and end points
            if len(search_history) > 0:
                ax.scatter([x_traj[0]], [y_traj[0]], [z_traj[0]], 
                          color='green', s=150, marker='o', label='Start', zorder=10)
                ax.scatter([x_traj[-1]], [y_traj[-1]], [z_traj[-1]], 
                          color='blue', s=150, marker='s', label='Best Found', zorder=10)
        else:
            x_traj = [point[0] for point in best_points]
            y_traj = [point[1] for point in best_points]
            z_traj = [point[2] for point in best_points]
            
            ax.plot(x_traj, y_traj, z_traj, color='white', linewidth=4, alpha=0.8, zorder=9)
            ax.plot(x_traj, y_traj, z_traj, color='red', linewidth=2.5, alpha=1.0, 
                    label='Search Trajectory', zorder=10)
            
            ax.scatter([x_traj[0]], [y_traj[0]], [z_traj[0]], 
                      color='lime', s=200, marker='o', label='Start', 
                      edgecolors='black', linewidth=2, zorder=15)
            ax.scatter([x_traj[-1]], [y_traj[-1]], [z_traj[-1]], 
                      color='cyan', s=200, marker='s', label='Best Found', 
                      edgecolors='black', linewidth=2, zorder=15)
            
            if len(best_points) > 2:
                ax.scatter(x_traj[1:-1], y_traj[1:-1], z_traj[1:-1], 
                          color='yellow', s=60, alpha=0.9, label='Search Points', 
                          edgecolors='black', linewidth=1, zorder=12)
        
        if len(best_points) >= 1 or len(search_history) > 0:
            legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                             fontsize=11, frameon=True, fancybox=True, 
                             shadow=True, framealpha=0.9)
            legend.get_frame().set_facecolor('white')
        
        ax.set_title(f'{func_name.capitalize()} Function - {algorithm_name} Trajectory', 
                    fontsize=16, pad=20, fontweight='bold')
        
        if x_range is not None and y_range is not None:
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    
    def plot_all_functions_grid(self, save_path=None):
        
        functions = [
            'sphere', 'ackley', 'rastrigin', 'rosenbrock', 'griewank',
            'schwefel', 'levy', 'michalewicz', 'zakharov'
        ]
        
        fig = plt.figure(figsize=(18, 12))
        
        for i, func_name in enumerate(functions):
            ax = fig.add_subplot(3, 3, i+1, projection='3d')
            
            # Get appropriate domain
            domain = get_function_domain(func_name)
            if func_name == 'schwefel':
                domain = (-100, 100)
            elif func_name == 'griewank':
                domain = (-20, 20)
            elif func_name == 'ackley':
                domain = (-10, 10)
            
            X, Y, Z = self.create_meshgrid(func_name, domain, domain)
            
            # Create surface plot
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)
            
            # Mark global optimum if known
            optimum_point, optimum_value = get_function_optimum(func_name)
            if optimum_point is not None:
                if (domain[0] <= optimum_point[0] <= domain[1] and 
                    domain[0] <= optimum_point[1] <= domain[1]):
                    ax.scatter([optimum_point[0]], [optimum_point[1]], [optimum_value],
                             color='red', s=50, marker='*', zorder=10)
            
            ax.set_title(f'{func_name.capitalize()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('X₁')
            ax.set_ylabel('X₂')
            ax.view_init(elev=30, azim=45)
        
        plt.suptitle('Optimization Functions - 3D Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_all_simulated_annealing_grid(self, results_dict, save_path=None):
        
        functions = [
            'sphere', 'ackley', 'rastrigin', 
            'rosenbrock', 'griewank', 'schwefel',
            'levy', 'michalewicz', 'zakharov'
        ]
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Simulated Annealing on All Benchmark Functions (iters=1000, seed=42)', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        for i, func_name in enumerate(functions):
            ax = fig.add_subplot(3, 3, i+1, projection='3d')
            
            vis_bounds = get_effective_visualization_bounds(func_name)
            X, Y, Z = self.create_meshgrid(func_name, vis_bounds, vis_bounds)
            
            surface = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8,
                                    linewidth=0, antialiased=True, 
                                    rcount=40, ccount=40, shade=True)
            
            if func_name in results_dict:
                result = results_dict[func_name]
                history = result['history']
                best_points = [point for point in history if point[3]]
                
                if len(best_points) >= 2:
                    x_traj = [point[0] for point in best_points]
                    y_traj = [point[1] for point in best_points]
                    z_traj = [point[2] for point in best_points]
                    
                    ax.plot(x_traj, y_traj, z_traj, color='white', linewidth=3, alpha=0.9)
                    ax.plot(x_traj, y_traj, z_traj, color='red', linewidth=2, alpha=1.0)
                    
                    ax.scatter([x_traj[0]], [y_traj[0]], [z_traj[0]], 
                              color='lime', s=80, marker='o', edgecolors='black', linewidth=1)
                    ax.scatter([x_traj[-1]], [y_traj[-1]], [z_traj[-1]], 
                              color='cyan', s=80, marker='s', edgecolors='black', linewidth=1)
                
                best_val = result['best_solution'].f
                ax.text2D(0.02, 0.98, f'best={best_val:.4f}', transform=ax.transAxes, 
                         fontsize=10, verticalalignment='top', 
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            optimum_point, optimum_value = get_function_optimum(func_name)
            if optimum_point is not None:
                if (vis_bounds[0] <= optimum_point[0] <= vis_bounds[1] and 
                    vis_bounds[0] <= optimum_point[1] <= vis_bounds[1]):
                    ax.scatter([optimum_point[0]], [optimum_point[1]], [optimum_value],
                             color='red', s=60, marker='*', edgecolors='black', linewidth=1)
            
            ax.set_title(f'{func_name.capitalize()}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('x₁', fontsize=10)
            ax.set_ylabel('x₂', fontsize=10)
            ax.set_zlabel('f(x)', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_alpha(0.1)
            ax.yaxis.pane.set_alpha(0.1)
            ax.zaxis.pane.set_alpha(0.1)
            
            ax.view_init(elev=30, azim=45)
            
            ax.locator_params(nbins=4)
            
            ax.set_xlim(vis_bounds[0], vis_bounds[1])
            ax.set_ylim(vis_bounds[0], vis_bounds[1])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig