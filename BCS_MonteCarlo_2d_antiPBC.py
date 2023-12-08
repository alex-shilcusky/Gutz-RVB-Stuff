# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:01:27 2023

@author: alexa
"""
import numpy as np 
import cmath
# import numba
import time
from matplotlib import pyplot as pl
from numpy import sin, cos, pi, linalg, sqrt
import itertools

L = 6
N = L*L
ones = np.ones(N)*1/2

t = 1
Delta = 5


''' site_indices and sites are neccessary global variables '''

sites = {} # sites[(n,m)] = index
site_indices = {} # site_indices[index] = (n,m)
index = 0

for y in range(L):
    for x in range(L):
        sites[(x,y)] = index
        site_indices[index] = (x,y)
        # print(index, neighbors)
        index += 1
        
def get_up_dn_sites(state): # function to get lists of up and down sites
        Nsite = len(state)
        sites_up = np.zeros(int(Nsite/2),dtype=int)
        sites_dn = np.zeros(int(Nsite/2),dtype=int)
        i=0
        j=0
        n=0
        while n < Nsite:
            if state[n] < 0:
                sites_dn[i] = n
                i+=1
            if state[n] > 0:
                sites_up[j] = n
                j+=1
            n+=1 
        return sites_up, sites_dn
def get_random_state(L):
    Nsite = L**2
    state = np.ones(Nsite)
    state[:Nsite//2] = -1
    state *= 0.5
    state = state[np.random.permutation(Nsite)]
    return state
def get_distance( i , j, site_indices):
    ni, mi = site_indices[i]
    nj, mj = site_indices[j]
    dn = nj - ni
    dm = mj - mi
    if abs(dn) > L/2:
        dn -= dn*L/abs(dn)
    if abs(dm) > L/2:
        dm -= dm*L/abs(dm)
    return sqrt(dn**2 + dm**2)

def get_nn1_array(L): # first nn array
    N = L**2
    neighbors_list = []
    nn = np.zeros(shape=(N,4), dtype=np.int16)
    for m in range(L):
        for n in range(L):
            neighbors = [ ((n+1)%L,m), (n,(m+1)%L), ((n-1)%L,m), (n, (m-1)%L) ]  # NN: x, y, -x, -y
            neighbors_list.append(neighbors)
    for i in range(N):
        neighbors = neighbors_list[i]
        for j in range(4):
            nn[i,j] = sites[neighbors[j]]
    return nn # NN: x, y, -x, -y



aPBC = 1

def get_H1a(N, t, Delta, nn): # nn pairing wavefunction w/ antiperiodic PBC
    H = np.zeros(shape=(2*N,2*N))
    for i in range(N):
        H[i,nn[i,0]] = -t 
        H[i,nn[i,1]] = -t
        H[i,nn[i,2]] = -t 
        H[i,nn[i,3]] = -t
        H[i+N,nn[i,0]+N] = t
        H[i+N, nn[i,1]+N] = t
        H[i+N,nn[i,2]+N] = t
        
        
        H[i,nn[i,0]+N] = Delta # 0 is for +x
        H[nn[i,0]+N,i] = Delta
        
        H[i,nn[i,1]+N] = -Delta # minus sign for y (1 is for +y)
        H[nn[i,1]+N,i] = -Delta
        
        H[i,nn[i,2]+N] = Delta # 2 is for -x
        H[nn[i,2]+N,i] = Delta
        
        H[i,nn[i,3]+N] = -Delta # minus sign for y gives delta(k) = delta(coskx - cosky)
        H[nn[i,3]+N,i] = -Delta     # 3 is for -y
        if aPBC: # False (true) for (anti) PBC
            if i>= L**2-L and i <= L**2-1:
                # print(i)
                H[i,nn[i,1]] *= -1
                H[i+N, nn[i,1]+N] *= -1 
                
                H[i,nn[i,1]+N] *= -1 
                H[nn[i,1]+N,i] *= -1
            if i>=0 and i<= L-1:
                # print(i)
                H[i,nn[i,3]] *= -1
                H[i+N, nn[i,3]+N] *= -1
                H[i,nn[i,3]+N] *= -1
                H[nn[i,3]+N,i] *= -1
    return H

     
def get_coeff2(U,state):
    N = len(state)
    Nsite = len(state)
    up, dn = get_up_dn_sites(state)
    Ne=len(up)
    A = np.zeros((Nsite,Nsite))
    for i in range(Nsite):
        for j in range(Ne):
            A[j,i] = U[up[j],i]
            A[j+Ne,i] = U[up[j]+N,i]
    return np.linalg.det(A)



def make_ising_state2D(L): # function to make an Ising state of size L 
    N = L**2
    state = []
    sgn = 1
    for i in range(N):
        if i%L ==0:
            sgn *= -1
        state.append(1*sgn)
        sgn *= -1
    return np.asarray(state)*0.5 # returns as [0.5, -0.5, ...]


J1 = 1
nn = get_nn1_array(L)


H = get_H1a(N, t, Delta, nn)
w, U = np.linalg.eigh(H)


x = make_ising_state2D(L)
print(get_coeff2(U,-x))


x = get_random_state(L)
print(get_coeff2(U,x))

def _psi(state):
    #state = state - 0.5*np.ones(len(state))
    #print(state)
    s = 0
    for i in range(len(state)):
        # for j in range(i+1, len(state)):
        for j in range(len(state)):
            if (i != j):
            # print(i,j)
                s -= 1/4*state[i]*state[j]/get_distance(i,j, site_indices)
    return np.exp(s)

def local_energy(state, coeff, Nsite, nn1,  J1): # get_eL for J1Heisenberg
    res = 0
    for i in range(Nsite):
        res += J1*state[i] * state[nn1[i,0]]
        res += J1*state[i] * state[nn1[i,1]]

    ssum = 0
    for i in range(Nsite):
        for k in range(0,2):
            if(state[i]*state[nn1[i,k]] < 0):
                state_new = state.copy()
                state_new[i] *= -1
                state_new[nn1[i,k]] *= -1
                c = get_coeff2(U, state_new)/coeff*J1
                if 0: # False (true) for (anti) PBC
                    if i>= L**2-L and i <= L**2-1:
                        c*= -1
                    if i>=0 and i<= L-1:
                        c*= -1
                ssum += c

    return res + 0.5*ssum

def metro(L, U, Delta, nn, plot, Nsample, Nskip):
    Nsite = L**2
    state = get_random_state(L)
    
    EE = np.zeros(Nsample)
    Etot = 0
    
    eL_list = np.zeros(Nsample)
    
    for i in range(Nsample):
        for j in range(Nskip):
            x = np.random.randint(low=0, high=Nsite)
            y = x
            while( state[y] * state[x] > 0):
                y = np.random.randint(low=0, high=Nsite)
            new_state = state.copy()
            new_state[x] *= -1
            new_state[y] *= -1
            
            # coeff_old = get_coeff2(U, state)*_psi(state)
            # coeff_new = get_coeff2(U, new_state)*_psi(new_state)
            coeff_old = get_coeff2(U, state)
            coeff_new = get_coeff2(U, new_state)
            
            if( np.random.random() < min(1.0, np.abs(coeff_new/coeff_old)**2)):
                state = new_state.copy()
                coeff_old = coeff_new
        eL = local_energy(state, coeff_old, Nsite, nn, J1)
        # print(state)
        # print('i=',i)
        Etot += eL
        EE[i] = Etot/(i+1)
        eL_list[i] = eL
        # print(EE[i])
        # EE = EE/Nsite
        
    if plot:
        it = np.linspace(0,Nsample, Nsample)

        pl.figure()
        pl.xlabel('step')
        pl.ylabel('Energy per site')
        pl.plot(it, EE/Nsite)
        # pl.ylabel('Energy ')
        # pl.plot(it, EE)
        pl.grid()
        pl.title(r'2D Square Lattice, L=%i $\Delta$=%.3f'%(L,Delta))
        pl.show()
        
    sigsq = 0
    # for eL in eL_list:
    for eL in EE:
        # print(eL)
        sigsq += (eL - EE[-1])**2
    sigsq /= Nsample
    # sigsq
    # Ef = EE[-1]
    sig = np.sqrt(sigsq)
    print(EE[-1]/Nsite, sig/Nsite)
    
    return EE[-1]/Nsite, sig/Nsite


t0 = time.time()



# E, sig = metro(L, U, Delta, nn, plot=True, Nsample=200,Nskip=3)

# t1 = time.time()
# print("Elapsed time: %.2f sec" % (t1 - t0))



# for sweep over deltas
if 0:
    deltas = np.linspace(0,5, 10)
    EE = np.zeros(len(deltas))
    error = np.zeros(len(deltas))
    # it = np.linspace(0,len(deltas), len(deltas))
    
    for i in range(len(deltas)):
        Delta = deltas[i]
        H = get_H1a(N,t,Delta,nn)
        (w,U) = np.linalg.eigh(H)
        E, sig = metro(L, U, Delta,nn, plot=False, Nsample=2000,Nskip=3)
        EE[i] = E
        error[i] = sig
        
    pl.figure()
    pl.errorbar(deltas[1:], EE[1:], yerr=error[1:], fmt = 'o', capsize=5)
    pl.title('2D Gutz BCS L=%i'%L)
    pl.xlabel('Delta')
    pl.ylabel('Energy per site')
    pl.grid()
    pl.show()
    