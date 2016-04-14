"""
container module for random utility fucntions that don't really fit anywhere else
"""

import numpy as np

# from http://people.duke.edu/~ccc14/pcfb/analysis.html


def bootstrap(data, statistic=np.mean, num_samples=10000, alpha=0.05):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    idx = np.random.randint(0, len(data), (num_samples, len(data)))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha / 2.0) * num_samples)], stat[int((1 - alpha / 2.0) * num_samples)])
