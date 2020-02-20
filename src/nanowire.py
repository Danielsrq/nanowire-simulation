#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 01:42:41 2019

@author: Domi
"""
from tqdm import tqdm
import kwant
import tinyarray as ta
import numpy as np
import scipy.sparse.linalg
from nanomagnet_field import rick_fourier
from transport_model import NISIN, barrier_region

# angstrom # 6.0583E-10 # 20E-9 # might need to change this.
lattice_constant_InAs = 200


class Nanowire:
    def __init__(
        self,
        width=5,
        length=5,
        dim=2,
        barrier_length=1,
        effective_mass=0.5,
        M=0.05,
        muSc=0.0,
        alpha_R=0.8,
        addedSinu=False,
        stagger_ratio=0.5,
        mu=0.3,
        delta=0.1,
        barrier=2.0,
        user_B=None,
        period= 16 # uses lattice size
    ):
        # Wire Physical Properties
        self.width = width
        self.length = length
        self.dim = dim
        self.barrier_length = barrier_length

        # Superconducting components
        self.t = 3.83 / (effective_mass*(lattice_constant_InAs**2))  # (hbar**2)/(2*effective_mass*electron_mass*(lattice_constant_InAs**2))# 0.5 / effective_mass
        self.M = M
        self.muSc = muSc
        self.alpha = alpha_R/lattice_constant_InAs
        self.addedSinu = addedSinu

        # Nanomagnet properties
        self.stagger_ratio = stagger_ratio
        self.user_B = user_B
        self.period = period

        # Previously hard-coded parameters
        self.mu = mu  # how is this different from muSc?
        self.delta = delta
        self.barrier = barrier
        
        # system
        self.system = None

        # System
        """self.system = NISIN(width=self.width, length=self.length, 
                                barrier_length=self.barrier_length, M=self.M,
                                addedSinu=self.addedSinu, 
                                stagger_ratio=self.stagger_ratio
                                )
        """

    def spectrum(self, bValues=np.linspace(0, 1.0, 201)):
        syst = NISIN(
            width=self.width,
            length=self.length,
            barrier_length=self.barrier_length,
        )
        self.system = syst
        energies = []
        critB = 0
        eigs_list = [] ## a list of eigenvalues and eigenvectors
        params = dict(
            muSc=self.muSc,
            mu=self.mu,
            Delta=self.delta,
            alpha=self.alpha,
            t=self.t,
            barrier=self.barrier,
            addedSinu=self.addedSinu,
            M=self.M,
            stagger_ratio=self.stagger_ratio,
            barrier_length=self.barrier_length,
            user_B=self.user_B,
            period=self.period
        )
        for i in tqdm(
            range(np.size(bValues)),
            desc="Spec: Length = %i, added? %r" % (self.length, self.addedSinu),
        ):
            b = bValues[i]
            params["B"] = b
            H = syst.hamiltonian_submatrix(sparse=True, params=params)
            H = H.tocsc()
            # k is the number of eigenvalues, and find them near sigma.
            eigs = scipy.sparse.linalg.eigsh(H, k=20, sigma=0)
            eigs_list.append(eigs)
            eigs = np.sort(eigs[0])
            energies.append(eigs)
            if critB == 0 and np.abs(eigs[10] - eigs[9]) / 2 < 1e-3:
                critB = b

        outcome = dict(B=bValues, E=energies, CritB=critB)
        return outcome, eigs_list

    def conductances(
        self,
        bValues=np.linspace(0, 1.0, 201),
        energies=[0.001 * i for i in range(-120, 120)],
    ):
        syst = NISIN(
            width=self.width,
            length=self.length,
            barrier_length=self.barrier_length
        )
        data = []
        critB = 0
        params = dict(
            muSc=self.muSc,
            mu=self.mu,
            Delta=self.delta,
            alpha=self.alpha,
            t=self.t,
            barrier=self.barrier,
            M=self.M,
            addedSinu=self.addedSinu,
            stagger_ratio=self.stagger_ratio,
            barrier_length=self.barrier_length,
            user_B=self.user_B,
            period=self.period
        )
        for i in tqdm(
            range(np.size(energies)),
            desc="Cond: Length = %i, added? %r" % (self.length, self.addedSinu),
        ):
            cond = []
            energy = energies[i]
            for b in tqdm(bValues, desc="bValues"):
                params["B"] = b
                smatrix = kwant.smatrix(syst, energy, params=params)
                conduct = (
                    smatrix.submatrix((0, 0), (0, 0)).shape[0]  # N
                    - smatrix.transmission((0, 0), (0, 0))  # R_ee
                    + smatrix.transmission((0, 1), (0, 0)) # maybe I need to update the 1 to a lattice_constant?
                )  # R_he
                cond.append(conduct)
                if (
                    np.isclose(energy, 0, rtol=1e-6)
                    and critB == 0
                    and np.abs(2 - conduct) < 0.01
                ):
                    critB = b
            data.append(cond)

        outcome = dict(B=bValues, BiasV=energies, Cond=data, CritB=critB)
        return outcome

    def plot(self):
        syst = NISIN(
            width=self.width,
            length=self.length,
            barrier_length=self.barrier_length,
        )

        length = self.length

        return kwant.plotter.plot(syst, show=False, unit='nn', site_size=0.20,
                                  site_color=lambda s: 'y'
                                  if barrier_region(
                                      s, self.barrier_length,
                                      length, self.width) else 'b')


def main():
    pass


if __name__ == "__main__":
    main()
