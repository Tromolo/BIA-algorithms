import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class TSPVisualizer:        
    def plot_tour(self, cities: List, route: List[int], title: str = "TSP Tour",
                  save_path: Optional[str] = None, show_labels: bool = True,
                  figsize: Tuple = (10, 10)) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        
        x_coords = [cities[i].x for i in route]
        y_coords = [cities[i].y for i in route]
        
        x_coords.append(cities[route[0]].x)
        y_coords.append(cities[route[0]].y)
        
        ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.7, label='Tour')
        
        city_x = [city.x for city in cities]
        city_y = [city.y for city in cities]
        ax.scatter(city_x, city_y, c='blue', s=100, zorder=5, edgecolors='black', linewidth=1.5)
        
        start_city = cities[route[0]]
        ax.scatter([start_city.x], [start_city.y], c='green', s=200, zorder=6,
                  marker='s', edgecolors='black', linewidth=2, label='Start')
        
        if show_labels:
            for i, city in enumerate(cities):
                ax.annotate(str(i), (city.x, city.y), fontsize=9, ha='center', va='center',
                           color='white', weight='bold', zorder=7)
        
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
