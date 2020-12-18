import numpy as np

def get_cor(y, yhat, zeros=True, normalize=False):
    if zeros:
        yhat = 2 * yhat - 1
        y = 2 * y - 1
        return np.inner(y, yhat) / yhat.shape[0] if normalize else np.inner(y, yhat)
    return np.inner(y, yhat) / yhat.shape[0] if normalize else np.inner(y, yhat)


def normalize(X, eps=1e-10):
    """
    X is n X d matrix. Make each coordinate in range [0,1].
    """
    n, d = X.shape
    Y = np.zeros_like(X, dtype=np.float32)
    for i in range(d):
        Y[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min() + eps)

    return Y


def get_gamma(wl_cor, h_cor, regret):
    e_wl = np.mean(np.asarray(wl_cor))
    e_h = np.mean(np.asarray(h_cor))
    return (e_wl + regret) / (e_h)