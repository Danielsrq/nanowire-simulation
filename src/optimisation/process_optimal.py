import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import sys
sys.path.append('../')
from nanowire import Nanowire

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['lines.linewidth'] = 1
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['mathtext.default'] = 'regular'


def plot_spectrum_data(params: Dict) -> dict:
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
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (7, 5)
    ax = fig.gca()
    ax.plot(spectrum_data["B"], spectrum_data["E"])
    ax.set_xlabel("Zeeman Field Strength [B]")
    ax.set_ylabel("Energies [t]")
    plt.savefig('mzm.pdf', dpi=1000)
    plt.show()


def plot_cost():
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    costs = np.load('./data/costs1.1417981541647708e-05.npy')
    actions = np.load('./data/action1.1417981541647708e-05.npy')
    print(costs[np.argmax(costs)])
    print(actions[np.argmax(costs)])
    print(actions[-1])
    print(costs[-1])
    plt.scatter([i+1 for i in range(len(costs))], costs, c='black', marker='x')
    plt.show()


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
x0['mu'] = 0.051000000000000024
x0['muSc'] = 0.01661
x0['barrier'] = 0.10300000000000001
spectrum_data = plot_spectrum_data(x0)

# plot_cost()
