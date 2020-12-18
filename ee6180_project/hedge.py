import math
import random
import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

ex = Experiment()
ex = initialise(ex)

def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i

    return len(probs) - 1

class Hedge():
    def __init__(self, n_expts, weights, eta=1):
        self.eta = eta
        self.n_expts = n_expts
        self.weights = weights
        self.bounds = (np.arange(self.n_expts) + 0.5)/(self.n_expts - 1)

    def initialize(self):
        self.weights = np.ones(self.n_expts) * (1./self.n_expts)

    def get_normalized_weights(self):
        probs = self.weights / np.sum(self.weights)
        return probs

    def choose_expt(self):
        probs = self.weights / np.sum(self.weights)
        return categorical_draw(probs)
    
    def play(self, x, y):
        # bounds = np.arange(17) + 0.5
        # bounds = bounds / 16.0 # normalizing bounds
        preds = np.asarray(x < self.bounds, dtype=np.float32)
        score = preds == y
        costs = 1 - score
        update = np.exp( -self.eta * costs)
        self.weights = self.weights * update
        self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, x):
        expt = self.choose_expt()
        y_pred = int(x < self.bounds[expt])
        return y_pred
        # return np.asarray([x < (expt + 0.5)/16.0]).astype(np.float32)


@ex.automain
def main(_run):
    params = edict(_run.config)
    eta = np.sqrt(np.log(params.n_expts) / params.T)
    hedge = Hedge(n_expts=params.n_expts, weights=None, eta=eta)

    # Create a dummy dataset
    X_ = np.random.randint(0, params.n_expts, size=params.T)/ (params.n_expts - 1)
    y_ = np.asarray(X_ < (5.5/(params.n_expts - 1)), dtype=np.float32)
    mask = np.random.random(size=params.T)
    mask = mask < params.threshold 
    y_ = (y_*mask).astype(np.float32)

    # Run hedge
    hedge.initialize()
    preds, chosen_expts = [], []
    for i in range(params.T):
        chosen_expts.append(hedge.choose_expt())
        preds.append(hedge.predict(X_[i])) # predict y_hat
        hedge.play(X_[i], y_[i]) # update weights
        if (i+1) % 1000 == 0:
            print(i+1, hedge.weights)


    plt.plot(hedge.weights)

    best_expt, best_score = 0, 0
    for i in range(params.n_expts):
        j = (i + 0.5)/(params.n_expts - 1)
        score = np.sum((X_ < j) == y_)/params.T
        if score > best_score:
            best_expt = i
            best_score = score

    print(best_score, best_expt)

    best_expt_preds = X_ < hedge.bounds[best_expt]
    best_expt_acc = best_expt_preds.flatten() == y_
    preds = np.asarray(preds)
    acc = preds.flatten() == y_
    cum_acc = np.cumsum(acc)
    cum_best_expt_acc = np.cumsum(best_expt_acc)
    regret = cum_best_expt_acc - cum_acc # regret
    plt.xlabel(r't$\rightarrow$')
    plt.ylabel(r'regret$\rightarrow$')
    plt.grid(which='both')
    plt.plot(regret, label='actual regret')
    plt.plot(np.sqrt(2*np.arange(params.T)*np.log(params.n_expts)), label='theoretical regret')
    plt.legend()


    trend_of_expts = np.zeros((params.n_expts, params.T))
    trend_of_expts[chosen_expts, np.arange(params.T)] = 1
    trend_of_expts = np.cumsum(trend_of_expts, axis=1)
    trend_of_expts = trend_of_expts / np.arange(1, params.T+1)
    plt.figure(figsize=(12, 9))
    plt.grid(which='both')
    plt.xlabel(r't$\rightarrow$')
    plt.ylabel(r'fraction$\rightarrow$')
    for i in range(trend_of_expts.shape[0]):
        plt.plot(trend_of_expts[i], label='expt_'+str(i))
        plt.legend()