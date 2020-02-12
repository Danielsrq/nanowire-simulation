import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from nanowire import Nanowire


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

    spectrum_data = nanowire.spectrum(bValues=np.linspace(0, params["b_max"], 81))
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (7, 5)
    ax = fig.gca()
    ax.plot(spectrum_data["B"], spectrum_data["E"])
    ax.set_xlabel("Zeeman Field Strength [B]")
    ax.set_ylabel("Energies [t]")
    print(spectrum_data["CritB"])
    plt.show()


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

simulation_single(params)
