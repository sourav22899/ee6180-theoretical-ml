from pathlib import Path
import numpy as np


def base_config():
    n_expts = 17
    T = 5000
    data_root = Path('./data/')
    log_root = Path('./logs/')
    expt_name = ''
    train_data = ''
    test_data = ''
    threshold = 1.0  # this is the fraction of data points whose labels are intact for testing hedge
    n_steps = 100  # for OCO
    n_iterations_gamma = 100
    D = 2
    K = 50
    n_wl = 100
    estimate_gamma_bool = False
    best_expt_bool = True


def ocr_config():
    expt_name = Path('ocr')
    train_data = Path('optdigits.tra')
    test_data = Path('optdigits.tes')
    n_wl = 64


def isolet_config():
    T = 7500
    expt_name = Path('isolet')
    train_data = Path('isolet1+2+3+4.data')
    test_data = Path('isolet5.data')
    n_iterations_gamma = 10
    n_wl = 617


named_configs = [ocr_config, isolet_config]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
