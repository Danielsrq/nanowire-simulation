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

s0 = np.identity(2)
sZ = np.array([[1., 0.], [0., -1.]])
sX = np.array([[0., 1.], [1., 0.]])
sY = np.array([[0., -1j], [1j, 0.]])

tauZ = ta.array(np.kron(sZ, s0))
tauX = ta.array(np.kron(sX, s0))
tauY = ta.array(np.kron(sY, s0))
sigZ = ta.array(np.kron(s0, sZ))
sigX = ta.array(np.kron(s0, sX))
sigY = ta.array(np.kron(s0, sY))
tauZsigX = ta.array(np.kron(sZ, sX))
tauZsigY = ta.array(np.kron(sZ, sY))
tauYsigY = ta.array(np.kron(sY, sY))

## Functions ##
def makeNISIN(width=5, noSections=5, barrierLen=1, M=0.05,
              addedSinu=False, isWhole=True
              ):
    length = 20*noSections + 2*barrierLen
    
    ## Define site Hopping and functions ##
    def hopX(site0, site1, t, alpha):
        return -t * tauZ + .5j * alpha * tauZsigY
    def hopY(site0, site1, t, alpha):
        return -t * tauZ - .5j * alpha * tauZsigX
        
    def sinuB(theta):
        return sigY*np.cos(theta) + sigX*np.sin(theta)
   
    def onsiteSc(site, muSc, t, B, Delta):
        if addedSinu: # Note 1 magnets per 20 sites - fixed distance of Magnets
            theta = 2 * np.pi * np.mod(site.pos[0] - 1,40)/40
            return (4 * t - muSc) * tauZ + B * sigX + Delta * tauX \
                    + M*sinuB(theta)
        else:
            return (4 * t - muSc) * tauZ + B * sigX + Delta * tauX
    def onsiteNormal(site, mu, t):
        return (4 * t - mu) * tauZ
    def onsiteBarrier(site, mu, t, barrier):
        return (4 * t - mu + barrier) * tauZ
    
    # On each site, electron and hole orbitals.
    lat = kwant.lattice.square(norbs=4) 
    syst = kwant.Builder()
        
    # S
    syst[(lat(i, j) for i in range(1,length-1) for j in range(width))] \
        = onsiteSc
        
    if isWhole:
        # I's
        syst[(lat(i, j) for i in range(barrierLen) for j in range(width))] = onsiteBarrier
        syst[(lat(i, j) for i in range(length-barrierLen, length)for j in range(width))] = onsiteBarrier
        
        # Hopping:
        syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopX
        syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopY
    
        # N's
        lead = kwant.Builder(kwant.TranslationalSymmetry((-1,0)),
                             conservation_law=-tauZ,
                             particle_hole=tauYsigY
                             )
        lead[(lat(0, j) for j in range(width))] = onsiteNormal
        lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopX
        lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopY
        
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())
    else:
        # Hopping:
        syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopX
        syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopY
    
    return syst.finalized()

## Objects ##
class Nanowire:
    def __init__(self, width=5, noSections=5, barrierLen=1, M=0.05,
                 addedSinu=False, dim=2):
        self.width = width
        self.noSections = noSections
        self.barrierLen = barrierLen
        self.addedSinu = addedSinu
        self.dim = dim
        self.M = M
        
    def spectrum(self, 
                 bValues=np.linspace(0, 1.0, 201)
                 ):        
        syst = makeNISIN(width=self.width, noSections=self.noSections, 
                         barrierLen=self.barrierLen, M=self.M,
                         addedSinu=self.AddedSinu, isWhole=False
                         )
        energies = []
        critB = 0
        params = dict(muSc=0., mu=.3, Delta=.1, alpha=.8, t=1., barrier=2.)
        for i in tqdm(range(np.size(bValues)), 
                      desc="Number of sections = %i" %(self.noSections)
                      ):
            b = bValues[i]
            params["B"] = b
            H = syst.hamiltonian_submatrix(sparse=True,  params=params)
            H = H.tocsc()
            eigs = scipy.sparse.linalg.eigsh(H, k=20, sigma=0)
            eigs = np.sort(eigs[0])
            energies.append(eigs)
            if critB==0 and np.abs(eigs[10] - eigs[9])/2 < 1e-3:
                critB = b
            
        outcome = dict(B=bValues, E=energies, CritB=critB)
        return outcome
    
    def conductances(self, 
                     bValues=np.linspace(0, 1.0, 201),
                     energies=[0.001 * i for i in range(-120, 120)]
                     ):
        syst = makeNISIN(width=self.width, noSections=self.noSections, 
                         barrierLen=self.barrierLen, M=self.M,
                         addedSinu=self.addedSinu, isWhole=True,
                         )
        data = []
        critB = 0
        params = dict(muSc=0., mu=.3, Delta=.1, alpha=.8, t=1., barrier=2.)
        for i in tqdm(range(np.size(energies)), 
                      desc="Number of sections = %i" %(self.noSections)
                      ):
            cond = []
            energy = energies[i]
            for b in bValues:
                params["B"] = b
                smatrix = kwant.smatrix(syst, energy, params=params)
                conduct = (
                        smatrix.submatrix((0, 0), (0, 0)).shape[0]  # N
                        - smatrix.transmission((0, 0), (0, 0))      # R_ee
                        + smatrix.transmission((0, 1), (0, 0)))     # R_he
                cond.append(conduct)
                if energy == 0 and critB == 0 and np.abs(2 - conduct) < 0.01:
                    critB = b
            data.append(cond)
            
        outcome = dict(B=bValues, BiasV=energies, Cond=data, CritB=critB)
        return outcome
    
    def phaseTransition(self,
                        bValues=np.linspace(0, 1., 501),
                        muValues=np.linspace(0, .5, 51)
                     ):
        syst = makeNISIN(width=self.width, noSections=self.noSections, 
                         barrierLen=self.barrierLen, M=self.M,
                         addedSinu=self.AddedSinu, isWhole=False
                         )
        criticalPoints = []
        params = dict(mu=.3, Delta=.1, alpha=.8, t=1.0, barrier=2.)
        for i in tqdm(range(np.size(muValues)),
                      desc="Number of sections = %i" %(self.noSections)
                      ):
            params["muSc"] = muValues[i]
            for b in bValues:
                params["B"] = b
                H = syst.hamiltonian_submatrix(sparse=True,  params=params)
                H = H.tocsc()
                eigs = scipy.sparse.linalg.eigsh(H, k=20, sigma=0)
                eigs = np.sort(eigs[0])
                if np.abs(eigs[9] - eigs[10])/2 < 1e-3:
                    criticalPoints.append(b)
#                    print("mu = %1.2f, b = %1.2f" %(muValues[i], b))
                    break
            else:
                continue
            
        outcome = dict(MuSc=muValues, CritB=criticalPoints)
        return outcome
    
    def phaseAid(self, 
                 bValues=np.linspace(.0, .3, 61),
                 muValues=np.linspace(.2, .5, 61)
                 ):
        syst = makeNISIN(width=self.width, noSections=self.noSections, 
                         barrierLen=self.barrierLen, M=self.M,
                         addedSinu=self.AddedSinu, isWhole=False
                         )
        energies0 = []
        energies1 = []
        params = dict(muSc=muValues[0], mu=.3, Delta=.1, alpha=.8, t=1., barrier=2.)
        for i in tqdm(range(np.size(bValues)), 
                      desc="Number of sections = %i" %(self.noSections)
                      ):
            b = bValues[i]
            params["B"] = b
            H = syst.hamiltonian_submatrix(sparse=True,  params=params)
            H = H.tocsc()
            eigs = scipy.sparse.linalg.eigsh(H, k=20, sigma=0)
            energies0.append(np.sort(eigs[0]))
            
        for i in tqdm(range(np.size(muValues)), 
                      desc="Number of sections = %i" %(self.noSections)
                      ):
            mu = muValues[i]
            params["muSc"] = mu
            H = syst.hamiltonian_submatrix(sparse=True,  params=params)
            H = H.tocsc()
            eigs = scipy.sparse.linalg.eigsh(H, k=20, sigma=0)
            energies1.append(np.sort(eigs[0]))

            
        outcome = dict(B=bValues, MuSc=muValues, Eb=energies0, Em=energies1)
        return outcome

def main():
    pass

if __name__ == '__main__':
    main()