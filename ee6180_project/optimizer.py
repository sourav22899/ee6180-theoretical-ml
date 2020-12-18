import numpy as np

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

ex = Experiment()
ex = initialise(ex)

class OnlineConvexOptimizer():
    """

    Online Gradient Descent according to 
    Elad Hazan. Introduction to online convex optimization. Foundations and Trends in Optimization, 2(3-4):157–325, 2016., Chapter 3.1

    """
    def __init__(self, D=2, gamma=0.02):
        self.D = D
        self.gamma = gamma
        self.G = 2./self.gamma
        self.iter = 0

    def initialize(self):
        self.iter = 0

    def project(self, x):
        """
        Project to [-1,1] i.e. the convex set K upon which are projecting.
        """
        if x < -1:
            return -1
        if x > 1:
            return 1
        return x
    
    def step(self, x, f):
        self.iter = self.iter + 1
        stepsize = self.D/(self.G*np.sqrt(self.iter))
        y = x - stepsize * f(1)
        x = self.project(y)
        return x


def f(x):
    return x * np.random.random()

@ex.automain
def main(_run):
    params = edict(_run.config)    
    oco = OnlineConvexOptimizer()
    x = 0
    for _ in range(params.n_steps):
        x_ = oco.step(x, f)
        print(oco.iter, x, f(x), x_)
        x = x_