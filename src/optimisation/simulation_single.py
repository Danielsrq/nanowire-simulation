import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from nanowire import Nanowire
import sys
sys.path.append('../')


def process_spectrum(E, B_step, cost_f, threshold=1E-6):
    E = np.array(E)
    # Costs for each EigenE
    costs = []
    # For each EigenE
    for i in range(len(E[0])):
        energy_single = E[:, i]
        # index of zero energy points for a given EigenE
        zeroE_indices = [i for i, j in enumerate(energy_single)
                         if np.abs(j) < threshold]
        # list of lists of consecutive indices
        clusters = [list(cluster)
                    for cluster in mit.consecutive_groups(zeroE_indices)]
        # array of arrays of (B_crit, B_width)
        cost = [cost_f(B_step * np.array([i[0], len(i)])) for i in clusters]
        if len(cost) == 0:
            cost = [0]
        # only interested in the maximum cost of each EigenE line
        cost = max(cost)
        costs.append(cost)
        # print (i)
        # print (energy_single)
        # print (zeroE_indices)
        # print (clusters)
        # print (cost)
        # print ("\n")
    final_cost = max(costs)
    # print (final_cost)
    return final_cost


def weighting(arr):
    """ test weighting function """
    # shld not be called
    if len(arr) != 2:
        print("array has len != 2")
        return 0
    else:
        Bcrit, B_width = arr[0], arr[1]
        return Bcrit + B_width


def simulation_single(params):
    nanowire = Nanowire(
        width=params["wire_width"],
        noMagnets=params["N"],
        effective_mass=params["effective_mass"],
        muSc=params["muSc"],
        alpha_R=params["alpha_R"],
        M=params["M"],
        addedSinu=params["added_sinusoid"],
        stagger_ratio=params["ratio"],
        mu=params["mu"],
        delta=params["delta"],
        barrier=params["barrier"],
    )
    spectrum_data = nanowire.spectrum(bValues=np.linspace(0, params["b_max"], 21))
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (7, 5)
    ax = fig.gca()
    ax.plot(spectrum_data["B"], spectrum_data["E"])
    ax.set_xlabel("Zeeman Field Strength [B]")
    ax.set_ylabel("Energies [t]")
    print(spectrum_data["CritB"])
    plt.show()
    return spectrum_data


params = dict()
params['wire_width'] = 7
params['N'] = 7
params['ratio'] = 0.5
params['M'] = 1
params['added_sinusoid'] = False
params['effective_mass'] = 0.023
params['alpha_R'] = 0.32
params['muSc'] = 0.01661
params['mu'] = 0.019
params['delta'] = 4.5E-05
params['barrier'] = 0.1
params['b_max'] = 1

# testing cost function
spectrum_data = simulation_single(params)
B, E = spectrum_data["B"], spectrum_data["E"]
B_step = B[1] - B[0]
cost = process_spectrum(E, B_step, weighting)
