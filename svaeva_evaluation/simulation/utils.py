import random
import string

import numpy as np


def wait_time_model(mean_wait_time, std_wait_time, num_samples=1, lb=0.5, ub=30):
    """
    Generate wait times for people messaging on an app based on a normal distribution.

    Parameters:
    - mean_wait_time (float): The mean wait time in seconds.
    - std_wait_time (float): The standard deviation of the wait time in seconds.
    - num_samples (int): The number of wait times to generate. Default is 1.

    Returns:
    - wait_times (float or array): The generated wait times.
    """
    wait_times = np.random.normal(mean_wait_time, std_wait_time, num_samples)
    # Trucate between lower bound and upper bound
    wait_times = np.clip(wait_times, lb, ub)
    return wait_times


def generate_random_string(length):
    """Generate a random string of the specified length."""
    letters = string.ascii_letters  # Contains all the letters (both lowercase and uppercase)
    return "".join(random.choice(letters) for _ in range(length))
