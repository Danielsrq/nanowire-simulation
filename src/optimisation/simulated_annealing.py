import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
import random
from nptyping import Array
from typing import Callable, List, Dict
import sys
sys.path.append('../')
from nanowire import Nanowire


def get_cost_from_spectrum(E: List[Array[np.float64]],
                           B_step: float,
                           cost_f: Callable[[List[float]], float],
                           threshold=1E-6) -> float:
    E = np.array(E)
    # Costs for each EigenE
    costs = []
    # For each EigenE
    for i in range(len(E[0])):
        energy_single = E[:, i]
        # List of indices of zero energy points for a given EigenE
        zeroE_indices = [i for i, j in enumerate(energy_single)
                         if np.abs(j) < threshold]
        # List of lists of consecutive indices
        clusters = [list(cluster)
                    for cluster in mit.consecutive_groups(zeroE_indices)]
        # Pass list of arrays of (B_crit, B_width) to cost
        cost = [cost_f(B_step * np.array([i[0], len(i)])) for i in clusters]
        if len(cost) == 0:
            cost = [0]
        # Get maximum cost of each EigenE line
        cost = max(cost)
        costs.append(cost)
    final_cost = max(costs)
    return final_cost


def weighting(arr: Array[float, int]) -> float:
    w1, w2 = 1, 1
    Bcrit, B_width = w1 * arr[0], w2 * arr[1]
    return Bcrit + B_width


def get_spectrum_data(params: Dict) -> dict:
    nanowire = Nanowire(
        width=params['wire_width'],
        noMagnets=params['N'],
        effective_mass=params['effective_mass'],
        muSc=params['muSc'],
        alpha_R=params['alpha_R'],
        M=params['M'],
        addedSinu=params['added_sinusoid'],
        stagger_ratio=params['ratio'],
        mu=params['mu'],
        delta=params['delta'],
        barrier=params['barrier'],
        user_B=params['user_B'],
    )
    spectrum_data = nanowire.spectrum(bValues=np.linspace(0, params['b_max'], 21))
    return spectrum_data


def acceptance_probability(old, new, T):
    """
    If the new cost value is larger than the old one, accept the new cost
    value with 100% probability. If the new cost value is smaller than the old,
    accept the new cost value with a probability equal to exp[(new - old) / T]
    T gets smaller at every iteration, i.e. we accept weaker cost values with
    lower probability
    """
    if new > old:
        a = 1
    else:
        a = np.exp((new - old) / T)
    return a


def simulated_annealing(x0: dict, x0_str: [str], T: float,
                        T_min: float, alpha: float):
    max_costs = []
    max_actions = []
    max_cost = 0.0
    x_new = x0.copy()
    while T > T_min:
        count = 0
        while(count < 100):
            spectrum_data = get_spectrum_data(x0)
            B, E = spectrum_data['B'], spectrum_data['E']
            B_step = B[1] - B[0]
            cost_new = get_cost_from_spectrum(E, B_step, weighting)
            print('x0: ', x0)
            print('x_new: ', x_new)
            print('new cost: ' + str(cost_new))
            ap = acceptance_probability(max_cost, cost_new, T)
            if ap > random.random():
                x0 = x_new.copy()
                max_cost = cost_new
                max_costs.append(max_cost)
                max_actions.append(x0)
            # Choose a parameter to change
            param = random.choice(x0_str)
            # Reset x_new to the latest accepted action
            x_new = x0.copy()
            # Make a move and make sure it is different from original
            new_move = random.choice([1E-6, -1E-6])
            x_new[param] += new_move
            count += 1
            print('max cost: ' + str(max_cost))
        np.save("sim_annealing/costs" + str(T), max_costs)
        np.save("sim_annealing/action" + str(T), max_actions)
        T = T * alpha


x0 = dict()
x0['wire_width'] = 7
x0['N'] = 7
x0['ratio'] = 0.5
x0['M'] = 1
x0['added_sinusoid'] = None
x0['effective_mass'] = 0.023
x0['alpha_R'] = 0.32
x0['delta'] = 4.5E-05
x0['b_max'] = 1
x0['user_B'] = None
x0['mu'] = 0.019
x0['muSc'] = 0.01661
x0['barrier'] = 0.1

x0_str = ['mu', 'muSc', 'barrier']
T = 1.0
T_min = 0.00001
alpha = 0.8
simulated_annealing(x0, x0_str, T, T_min, alpha)
