#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

from utils import *
from hedge import Hedge
from optimizer import OnlineConvexOptimizer
from booster import OnlineBooster

ex = Experiment()
ex = initialise(ex)

def estimate_gamma(X, y, T, n_expts, wl, n_iterations):
    n, d = X.shape
    wl_cor, h_cor, gammas = [], [], []
    theoretical_regret = np.sqrt(2 * T * np.log(n_expts))
    final_weights = np.zeros((d, n_expts))  # a 64*17 dim matrix to store the final weights of 64 weak learners
    for wl_no in tqdm(range(d)):  # choose the dimension along which you require the weak learner
        assert X.max() <= 1
        assert X.min() >= 0
        cum_weights = np.zeros(n_expts)
        for _ in range(n_iterations):
            start = np.random.choice(n - T - 1)
            X_wl, y_sampled = X[start:start + T, wl_no], y[start:start + T]
            X_lr = X[start:start + T]
            wl.initialize()
            preds = []
            for i in range(T):
                preds.append(wl.predict(X_wl[i]))  # predict y_hat
                wl.play(X_wl[i], y_sampled[i])  # update weights

            preds = np.asarray(preds).flatten()
            wl_cor.append(get_cor(y_sampled, preds))
            # print(X_lr.shape)
            # import pdb; pdb.set_trace()
            clf = LogisticRegression(random_state=0, fit_intercept=True).fit(X_lr, y_sampled)
            yhat = clf.predict(X_lr)
            h_cor.append(get_cor(y_sampled, yhat))
            cum_weights += wl.weights
        final_weights[wl_no] = cum_weights / n_iterations
        gammas.append(get_gamma(wl_cor, h_cor, theoretical_regret))
    return final_weights, gammas


def load_weights(final_weights, n_expts=17, eta=None, best_expt=-1, n_wl=100):
    weak_learners = dict()
    if best_expt == -1:
        n_wl = final_weights.shape[0]
        for i in range(n_wl):  # change back to final_weights.shape[0] for warm starts
            weak_learners[i] = Hedge(n_expts=n_expts, weights=final_weights[i], eta=eta)
    else:
        for i in range(n_wl):  # change back to final_weights.shape[0] for warm starts
            weak_learners[i] = Hedge(n_expts=n_expts, weights=final_weights[best_expt], eta=eta)

    return weak_learners

@ex.automain
def main(_run):
    params = edict(_run.config)

    train_data_path = params.data_root / params.expt_name / params.train_data
    test_data_path = params.data_root / params.expt_name / params.test_data
    data_tes = pd.read_csv(train_data_path, header=None)
    data_tra = pd.read_csv(test_data_path, header=None)
    data = pd.concat([data_tra, data_tes])

    data_sample = data.sample(frac=1)
    X, y = data_sample.to_numpy()[:, :-1], data_sample.to_numpy()[:, -1]
    print(X.shape, y.shape)

    X = normalize(X)  # simple normalization
    y = y % 2
    clf = LogisticRegression(random_state=0, fit_intercept=True).fit(X, y)
    yhat = clf.predict(X)
    print('correlation:', get_cor(y, yhat, normalize=True))

    # Plan : For OCR dataset, we will have 64 online weak learners corresponding to each coordinate of X. These weak learners are
    # Hedge algorithms. Each instance of Hedge has access to 17 experts corresponding to the fact that each
    # coordinate of X can take 17 values (0-16). Now, we will observe an data point (x,y) and we are interested to
    # choose a threshold above which the weak learner would predict x.

    # Estimating gamma

    # Sample T = 5000 and pass them in arbitrary order in get empirical estimation of expectation.  We use $R_W(
    # T)=\sqrt{T\log N}$. Then calculate $\gamma$ as: $$\gamma\leq\frac{E[<W(x_t),y_t>]+R_W}{E[<h(x_t),y_t>]}$$

    eta = np.sqrt(np.log(params.n_expts) / params.T)
    wl = Hedge(n_expts=params.n_expts, weights=None, eta=eta)
    wl.initialize()
    if params.estimate_gamma_bool:
        from warnings import simplefilter
        simplefilter(action='ignore')
        # Above lines are commented to suppress Convergence Warnings of lbfgs in sklearn.
        final_weights, gammas = estimate_gamma(X=X, y=y, T=params.T, n_expts=params.n_expts, wl=wl,
                                               n_iterations=params.n_iterations_gamma)

        gammas = np.asarray(gammas)
        save_path_final_weights = params.log_root / params.expt_name / 'final_weights'
        save_path_gamma = params.log_root / params.expt_name / 'gammas'
        np.save(save_path_final_weights, final_weights)
        np.save(save_path_gamma, gammas)
        print(gammas.max(), gammas.min(), gammas.mean())

    load_path_final_weights = params.log_root / params.expt_name / 'final_weights.npy'
    load_path_gamma = params.log_root / params.expt_name / 'gammas.npy'
    gammas = np.load(load_path_gamma)
    final_weights = np.load(load_path_final_weights)

    import pdb; pdb.set_trace()
    if params.best_expt_bool:
        best_expt = gammas.argmax()
        gamma = gammas[best_expt]
    else:
        best_expt = -1
        gamma = gammas.min()

    n, d = X.shape
    results = np.zeros((params.K, 5))
    theoretical_regret = np.sqrt(2 * params.T * np.log(params.n_expts))
    t1 = theoretical_regret / (gamma * params.T)
    for k in range(params.K):
        start = np.random.choice(n - params.T - 1)
        X_sampled, y_sampled = X[start:start + params.T], y[start:start + params.T]

        # Keep multiple instance of best weak learner instead of keeping one wrt each dim else best_expt = -1
        oco = OnlineConvexOptimizer(gamma=gamma)
        weak_learners = load_weights(final_weights, n_expts=params.n_expts, eta=eta, best_expt=best_expt, n_wl=params.n_wl)
        oco.initialize()
        booster = OnlineBooster(weak_learners=weak_learners, oco=oco, gamma=gamma, T=params.T, best_wl=best_expt)
        yhat_list = booster.run(X_sampled, y_sampled)

        clf = LogisticRegression(random_state=0, fit_intercept=True).fit(X_sampled, y_sampled)
        yhat_lr = clf.predict(X_sampled)
        h_star_cor = get_cor(y_sampled, yhat_lr, normalize=True)
        y_temp = np.sign(y_sampled - 0.5)
        pred_cor = get_cor(y_temp, yhat_list, zeros=False, normalize=True)
        results[k, 0] = h_star_cor
        results[k, 1] = pred_cor
        results[k, 2] = max(booster.grads)
        G = min(2. / gamma, max(booster.grads))
        t2 = (1.5 * G * np.sqrt(booster.N)) / booster.N
        results[k, 3] = t1
        results[k, 4] = t2
        print(h_star_cor, pred_cor, max(booster.grads))

    expected_regret = results[:, 0] - results[:, 1]
    regret_avg = np.mean(expected_regret)
    regret_std_dev = np.std(expected_regret)

    expected_upper_bound = results[:, 3] + results[:, 4]
    upper_bound_avg = np.mean(expected_upper_bound)
    upper_bound_std_dev = np.std(expected_upper_bound)

    save_path_results = params.log_root / params.expt_name / 'results'
    np.save(save_path_results, results)
    import pdb;
    pdb.set_trace()

# ## **New dataset: ISOLET**

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression

# data_tes = pd.read_csv('/content/drive/MyDrive/ee6180_project/isolet1+2+3+4.data', header=None)
# data_tra = pd.read_csv('/content/drive/MyDrive/ee6180_project/isolet5.data', header=None)
# data = pd.concat([data_tra, data_tes])
# data = data.sample(frac=1)
# data.head()


# data_sample = data.sample(frac = 1)
# X, y = data_sample.to_numpy()[:,:-1], data_sample.to_numpy()[:,-1]
# print(X.shape, y.shape)


# X = normalize(X)
# y = y % 2


# clf = LogisticRegression(random_state=0 ,fit_intercept=True).fit(X, y) 
# yhat = clf.predict(X)
# get_cor(y, yhat, normalize=True)


# n_expts = 17
# T = 7500
# eta = np.sqrt(np.log(n_expts)/T)


# wl = Hedge(n_expts=n_expts, weights=None, eta=eta)
# wl.initialize()
# wl.weights


# n_iterations = 10
# final_weights, gammas = estimate_gamma(X = X, y = y, T = T, n_expts = n_expts, wl = wl, n_iterations = n_iterations)
# gammas = np.asarray(gammas)
# np.save('/content/drive/MyDrive/ee6180_project/final_weights_isolet', final_weights)
# np.save('/content/drive/MyDrive/ee6180_project/gammas_isolet', gammas)
# print(gammas.max(), gammas.min(), gammas.mean())


# gammas = np.load('/content/drive/MyDrive/ee6180_project/gammas_isolet.npy')
# final_weights = np.load('/content/drive/MyDrive/ee6180_project/final_weights_isolet.npy')


# gamma = gammas.max()
# best_expt = gammas.argmax()
# best_expt

# gamma

# n, d = X.shape
# d

# K = 4
# n_wl = d # using 617 instances of weak learners
# results = np.zeros((K, 2))
# for k in range(K):
#   start = np.random.choice(n - T - 1)
#   X_sampled, y_sampled = X[start:start + T], y[start:start + T]
#   # Keep multiple instance of best weak learner instead of keeping one wrt each dim else best_expt = -1
#   oco = OnlineConvexOptimizer(gamma = gamma)
#   weak_learners = load_weights(final_weights, best_expt=best_expt, n_wl = n_wl) 
#   oco.initialize()
#   booster = OnlineBooster(weak_learners = weak_learners, oco = oco, gamma = gamma, T = T, best_wl=best_expt)
#   yhat_list = booster.run(X_sampled, y_sampled)

#   clf = LogisticRegression(random_state=0 ,fit_intercept=True).fit(X_sampled, y_sampled)
#   yhat_lr = clf.predict(X_sampled)
#   h_star_cor = get_cor(y_sampled, yhat_lr, normalize=True)
#   y_temp = np.sign(y_sampled - 0.5)
#   pred_cor = get_cor(y_temp, yhat_list, zeros=False, normalize=True)
#   results[k, 0] = h_star_cor
#   results[k, 1] = pred_cor
#   print(h_star_cor, pred_cor)


# G = 2./gamma
# D = 2
# N = X.shape[1] # change back to X.shape[1]


# G = min(G, max(booster.grads))


# t1 = (theoretical_regret)/(gamma * T)
# t2 = (1.5 * G * np.sqrt(N))/N
# t1 + t2

# t1

# t2
