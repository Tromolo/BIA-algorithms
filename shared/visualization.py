import numpy as np
import matplotlib.pyplot as plt
from .functions import Function, get_function_domain, get_function_optimum


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
        
        # Create surface plot with custom colormap
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                                linewidth=0, antialiased=True)
        
        # Add contour lines at the base
        ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), 
                  cmap='viridis', alpha=0.6)
        
        # Mark global optimum if known and requested
        if show_optimum:
            optimum_point, optimum_value = get_function_optimum(func_name)
            if optimum_point is not None:
                # Check if optimum is within the plotted range
                if (x_range[0] <= optimum_point[0] <= x_range[1] and 
                    y_range[0] <= optimum_point[1] <= y_range[1]):
                    ax.scatter([optimum_point[0]], [optimum_point[1]], [optimum_value],
                             color='red', s=100, marker='*')
        
        # Customize plot
        ax.set_xlabel('X₁', fontsize=12)
        ax.set_ylabel('X₂', fontsize=12)
        ax.set_zlabel(f'{func_name.capitalize()} Function Value', fontsize=12)
        ax.set_title(f'{func_name.capitalize()} Function - 3D Surface', fontsize=14, pad=20)
        
        # Add colorbar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=30)
        
        
        # Optimize viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
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
            
            # Plot trajectory
            ax.plot(x_traj, y_traj, z_traj, 'r-', linewidth=3, alpha=0.9, 
                    label='Search Trajectory')
            
            # Mark start and end points
            ax.scatter([x_traj[0]], [y_traj[0]], [z_traj[0]], 
                      color='green', s=150, marker='o', label='Start', zorder=10)
            ax.scatter([x_traj[-1]], [y_traj[-1]], [z_traj[-1]], 
                      color='blue', s=150, marker='s', label='Best Found', zorder=10)
            
            # Mark intermediate points
            if len(best_points) > 2:
                ax.scatter(x_traj[1:-1], y_traj[1:-1], z_traj[1:-1], 
                          color='orange', s=50, alpha=0.8, label='Search Points', zorder=8)
        
        # Only add legend if we have trajectory data
        if len(best_points) >= 1 or len(search_history) > 0:
            ax.legend()
        ax.set_title(f'{func_name.capitalize()} Function - {algorithm_name} Trajectory', fontsize=14, pad=20)
        
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