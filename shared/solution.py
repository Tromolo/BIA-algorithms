import numpy as np


class Solution:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.parameters = np.zeros(self.dimension)  # solution parameters
        self.f = np.inf  # objective function evaluation
    
    def generate_random(self):
        self.parameters = np.random.uniform(self.lB, self.uB, self.dimension)
    
    def evaluate(self, function):
        self.f = function.evaluate(self.parameters)
    
    def copy(self):
        new_solution = Solution(self.dimension, self.lB, self.uB)
        new_solution.parameters = self.parameters.copy()
        new_solution.f = self.f
        return new_solution
    
    def __str__(self):
        params_str = np.array2string(self.parameters, precision=4, suppress_small=True)
        return f"Solution(f={self.f:.6f}, params={params_str})"
    
    def __repr__(self):
        return self.__str__()
