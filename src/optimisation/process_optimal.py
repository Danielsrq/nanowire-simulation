import kwant
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from process_field import interpolate_field
import sys
sys.path.append('../')
from nanowire import Nanowire

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['mathtext.default'] = 'regular'


def make_wire_obj(params: Dict) -> dict:
    nanowire = Nanowire(
        width=params['wire_width'],
        length=params['N'],
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
        period=params['period'],
        )
    return nanowire


def plot_fullwf(kwant_syst, eigslist, Bslice, wf_index):
    # eigslist = np.array(eigslist)
    print("slice is", Bslice)
    Eigval, Eigvec = eigslist[Bslice]
    wf0 = abs(Eigvec[0::4, wf_index])**2 
    wf1 = abs(Eigvec[1::4, wf_index])**2 
    wf2 = abs(Eigvec[2::4, wf_index])**2 
    wf3 = abs(Eigvec[3::4, wf_index])**2 
    total_wf = wf0 + wf1 + wf2 + wf3
    up_wf = wf0 + wf3
    down_wf = wf1 + wf2
    print("Plotting wave function with index", wf_index)
    print("energy:", Eigval[wf_index])
    # kwant.plotter.map(kwant_syst, total_wf)
    kwant.plot(syst, site_color=total_wf,
               site_size=0.5, hop_lw=0, lead_site_symbol='s', colorbar=False, cmap='gist_heat_r')
    plt.savefig('fullwf.pdf', dpi=1000)
    kwant.plot(syst, site_color=up_wf,
               site_size=0.5, hop_lw=0, lead_site_symbol='s', colorbar=False, cmap='gist_heat_r')
    # plt.savefig('upwf.pdf', dpi=1000)
    kwant.plot(syst, site_color=down_wf,
               site_size=0.5, hop_lw=0, lead_site_symbol='s', colorbar=False, cmap='gist_heat_r')
    # plt.savefig('downwf.pdf', dpi=1000)


def plot_spectrum_data(params: Dict) -> dict:
    nanowire = Nanowire(
        width=params['wire_width'],
        length=params['N'],
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
        period=params['period'],
    )
    spectrum_data, eigs_list = nanowire.spectrum(bValues=np.linspace(0, params['b_max'], 31))
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (7, 5)
    ax = fig.gca()
    ax.plot(spectrum_data["B"], spectrum_data["E"])
    ax.set_xlabel("Zeeman Field Strength [B]")
    ax.set_ylabel("Energies [t]")
    plt.savefig('mzm.pdf', dpi=1000)
    plt.show()
    return spectrum_data, eigs_list


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


# 4t = 0.016600eV
# k_B (Boltzmann) = 8.617E-5 eV K^-1 ## at 10mK, kT ~ 8.6E-7
x0 = dict()
x0['wire_width'] = 6
x0['N'] = 80
x0['ratio'] = 0.5
x0['M'] = 0.5
x0['added_sinusoid'] = None
x0['effective_mass'] = 0.023
x0['alpha_R'] = 0.32
x0['delta'] = 190E-6  # 4.5E-05
x0['b_max'] = 2
x0['user_B'] = interpolate_field('./data/strayfield_halbach_100_60.ovf',
                                 nx=(80j), ny=(6j))
x0['mu'] = 0.019
x0['muSc'] = 0.0165
x0['barrier'] = 0.1
x0['period'] = 4000  # in Angstroms

nanowire = make_wire_obj(x0)
spectrum_data, eigs_list = nanowire.spectrum(bValues=np.linspace(0,
                                                                 x0['b_max'],
                                                                 31))
syst = nanowire.system
B, E = spectrum_data['B'], spectrum_data['E']

fig = plt.figure()
plt.rcParams["figure.figsize"] = (7, 5)
ax = fig.gca()
ax.plot(B, E)
ax.set_xlabel("Zeeman Field Strength [B]")
ax.set_ylabel("Energies [t]")
# plt.savefig('spectrum_idealsine.pdf', dpi=1000)
plt.show()

# Bstep = B[1]- B[0]
# Bslice = int(1 / Bstep)
# plot_fullwf(syst, eigs_list, Bslice, 0)   


# plot_cost()
