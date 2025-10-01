import numpy as np


class Function:
    def __init__(self, name):
        self.name = name
    
    def evaluate(self, params):
        if self.name == 'sphere':
            return self.sphere(params)
        elif self.name == 'ackley':
            return self.ackley(params)
        elif self.name == 'rastrigin':
            return self.rastrigin(params)
        elif self.name == 'rosenbrock':
            return self.rosenbrock(params)
        elif self.name == 'griewank':
            return self.griewank(params)
        elif self.name == 'schwefel':
            return self.schwefel(params)
        elif self.name == 'levy':
            return self.levy(params)
        elif self.name == 'michalewicz':
            return self.michalewicz(params)
        elif self.name == 'zakharov':
            return self.zakharov(params)
        else:
            raise ValueError(f"Unknown function: {self.name}")
    
    def sphere(self, params):        
        sum_val = 0
        for p in params:
            sum_val += p**2
        return sum_val
    
    def ackley(self, params, a=20, b=0.2, c=2*np.pi):        
        d = len(params)
        sum_sq = sum(p**2 for p in params)
        sum_cos = sum(np.cos(c * p) for p in params)
        
        term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        
        return term1 + term2 + a + np.exp(1)
    
    def rastrigin(self, params, A=10):
        n = len(params)
        sum_val = 0
        for p in params:
            sum_val += p**2 - A * np.cos(2 * np.pi * p)
        return A * n + sum_val
    
    def rosenbrock(self, params):
        sum_val = 0
        for i in range(len(params) - 1):
            sum_val += 100 * (params[i+1] - params[i]**2)**2 + (1 - params[i])**2
        return sum_val
    
    def griewank(self, params):        
        sum_sq = sum(p**2 for p in params)
        
        prod_cos = 1
        for i, p in enumerate(params):
            prod_cos *= np.cos(p / np.sqrt(i + 1))
        
        return sum_sq / 4000 - prod_cos + 1
    
    def schwefel(self, params):        
        n = len(params)
        sum_val = 0
        for p in params:
            sum_val += p * np.sin(np.sqrt(np.abs(p)))
        return 418.9829 * n - sum_val
    
    def levy(self, params):        
        w = [1 + (p - 1) / 4 for p in params]
        
        term1 = np.sin(np.pi * w[0])**2
        
        term2 = 0
        for i in range(len(w) - 1):
            term2 += (w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2)
        
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        
        return term1 + term2 + term3
    
    def michalewicz(self, params, m=10):        
        sum_val = 0
        for i, p in enumerate(params):
            sum_val += np.sin(p) * np.sin((i + 1) * p**2 / np.pi)**(2*m)
        return -sum_val
    
    def zakharov(self, params):        
        sum1 = sum(p**2 for p in params)
        sum2 = sum(0.5 * (i + 1) * p for i, p in enumerate(params))
        
        return sum1 + sum2**2 + sum2**4


def get_function_domain(func_name):
    """
    Reference: https://www.sfu.ca/~ssurjano/optimization.html
    """
    domains = {
        # Many Local Minima functions
        'ackley': (-32.768, 32.768),      # Ackley Function
        'griewank': (-600, 600),          # Griewank Function  
        'levy': (-10, 10),                # Levy Function
        'rastrigin': (-5.12, 5.12),       # Rastrigin Function
        'schwefel': (-500, 500),          # Schwefel Function
        
        # Bowl-Shaped functions
        'sphere': (-5.12, 5.12),          # Sphere Function
        
        # Plate-Shaped functions
        'zakharov': (-10, 10),             # Zakharov Function
        
        # Valley-Shaped functions
        'rosenbrock': (-2.048, 2.048),    # Rosenbrock Function
        
        # Steep Ridges/Drops functions
        'michalewicz': (0, np.pi)         # Michalewicz Function (0 to π)
    }
    return domains.get(func_name, (-10, 10))


def get_all_functions():
    return [
        'sphere', 'ackley', 'rastrigin', 'rosenbrock', 'griewank',
        'schwefel', 'levy', 'michalewicz', 'zakharov'
    ]


def get_function_categories():
    return {
        'Many Local Minima': ['ackley', 'griewank', 'levy', 'rastrigin', 'schwefel'],
        'Bowl-Shaped': ['sphere'], 
        'Plate-Shaped': ['zakharov'],
        'Valley-Shaped': ['rosenbrock'],
        'Steep Ridges/Drops': ['michalewicz']
    }


def get_effective_visualization_bounds(func_name):
    bounds = get_function_domain(func_name)
    
    adjustments = {
        'griewank': (-20, 20),
        'schwefel': (-400, 400), 
        'ackley': (-30, 30)
    }
    
    return adjustments.get(func_name, bounds)

def get_visualization_bounds(func_name):
    return get_effective_visualization_bounds(func_name)


def get_function_optimum(func_name, dim=2):
    optima = {
        'sphere': (np.zeros(dim), 0.0),
        'ackley': (np.zeros(dim), 0.0),
        'rastrigin': (np.zeros(dim), 0.0),
        'rosenbrock': (np.ones(dim), 0.0),
        'griewank': (np.zeros(dim), 0.0),
        'schwefel': (np.full(dim, 420.9687), 0.0),
        'levy': (np.ones(dim), 0.0),
        'michalewicz': (None, None),
        'zakharov': (np.zeros(dim), 0.0)
    }
    return optima.get(func_name, (None, None))
