import numpy as np
import random
from tqdm import tqdm


class OnlineBooster():
    """
    weak_learners: A dictionary of dictionary of N weak learners.
    T: time horizon
    gamma: AWOL parameter, in our case, it is set as 0.02
    oco: online convex optimizer (we use OGD)
    """

    def __init__(self, T=5000, gamma=0.02, weak_learners=None, oco=None, best_wl=-1):
        self.weak_learners = weak_learners
        self.T = T
        self.N = len(weak_learners)
        self.gamma = gamma
        self.oco = oco
        self.grads = []
        self.best_wl = best_wl

    def weak_learners_initialize(self):
        for algo in self.weak_learners:
            self.weak_learners[algo]['algo'].initialize()

    def randomized_project(self, x):
        if np.abs(x) >= 1:
            return np.sign(x)
        p1 = 0.5 * (1 + x)
        p2 = 0.5 * (1 - x)
        z = np.random.choice(np.asarray([1, -1]), p=[p1, p2])
        return z

    def randomized_label(self, y, p):
        # y in {0,1}
        z = random.random()
        if z < 0.5 * (1 + p):
            return y
        return 1 - y

    def booster_predict(self, x):
        preds = []
        for i in range(self.N):
            x_tilda = x[self.weak_learners[i]['idx']]
            p = self.weak_learners[i]['algo'].predict(x_tilda)
            p = np.sign(p - 0.5)  # To make predictions in {-1,+1}.
            preds.append(p)

        yhat = np.asarray(preds).mean() / self.gamma
        yhat = self.randomized_project(yhat)
        return yhat

    def update(self, x, y):
        """
        x: d-dim input, y in {0,1}
        """
        for i in range(self.N):
            if i == 0:
                p_ti = 0.0
            else:
                p_ti = self.oco.step(p_ti, l_ti)  # Not sure here

            x_tilda = x[self.weak_learners[i]['idx']]
            W_xt = np.sign(self.weak_learners[i]['algo'].predict(x_tilda) - 0.5)  # To make W(x_t) in {-1,+1}
            l_ti = lambda t: t * (((W_xt * y) / self.gamma) - 1)  # Not sure here also
            self.grads.append(np.abs(l_ti(1)))  # To compute G = max |grad(f(x))|
            y_random = self.randomized_label(y, p_ti)
            self.weak_learners[i]['algo'].play(x_tilda, y_random)

    def run(self, X, y):
        yhat_list = []
        for t in tqdm(range(self.T)):
            self.oco.initialize()
            xt, yt = X[t], y[t]
            yhat = self.booster_predict(xt)
            yhat_list.append(yhat)
            self.update(xt, yt)

        return np.asarray(yhat_list)
