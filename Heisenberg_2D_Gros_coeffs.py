# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:49:59 2023

@author: alexa
"""
import numpy as np 
import cmath
import numba
import time
from matplotlib import pyplot as pl
import matplotlib.patches as patches
from numpy import sin, cos, pi, linalg, sqrt
import itertools


t0 = time.time()



L = 4
N = L*L


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
        
def get_rij( i , j):
    xi, yi = site_indices[i]
    xj, yj = site_indices[j]
    xij = xj - xi
    yij = yj - yi
    if xij > L/2:
        xij -= xij*L/(xij)
    if xij < - L/2:
        xij += xij*L/(xij)
    if yij > L/2:
        yij -= yij*L/(yij)
    if yij < - L/2:
        yij += yij*L/(yij)
    return np.asarray([xij, yij])

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

def swap_spins(state, i, j):
    new_state = state.copy()
    si = state[i]
    sj = state[j]
    new_state[i] = sj
    new_state[j] = si
    return new_state

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

def get_klist(L):
    kk = []
    for nx in range(0,L):
        for ny in range(0,L):
            kx = 2*pi/L*nx
            ky = 2*pi/L*ny + pi/L
            kvec = np.asarray([kx,ky])
            kk.append(kvec)
    return kk

kk = get_klist(L)

if 0: # if you want to plot the 1st BZ and the k points 
    #   we use in the discrete FT
    rect = patches.Rectangle((-1,-1), 
                             2, 2, color='lightcoral')
    
    pl.figure()
    pl.axes()
    pl.gca().add_patch(rect)
    pl.gca().set(xlim=(-1.4,1.4),ylim=(-1.4,1.4))
    pl.grid()
    for k in kk:
            pl.plot(k[0]/pi, k[1]/pi, 'ko')
    pl.xlabel('$k_x/\pi$',fontsize=14)
    pl.ylabel('$k_y/\pi$',fontsize=14)
    pl.show()
    
    
def get_Delta_k(Delta, k):
    kx = k[0]
    ky = k[1]
    return 2*Delta*(cos(kx) - cos(ky))
    # return Delta


def get_xi_k(k):
    kx = k[0]
    ky = k[1]
    return -2*t*(cos(kx) + cos(ky))

def get_ak(k):
    Delta_k = get_Delta_k(Delta,k)
    xi_k = get_xi_k(k)
    # if xi_k == 0:
    #     return 1
    # if Delta_k == 0:
    #     return 0
    return Delta_k/(xi_k + np.sqrt( xi_k**2 + Delta_k**2 ))
    # return Delta_k/(xi_k + np.sqrt( abs(xi_k)**2 + abs(Delta_k)**2 ))
    
def get_a(r):
    s = 0
    for k in kk:
        s += get_ak(k)*cos(np.inner(r,k))
        # s += get_ak(k)*exp(1j*(np.inner(r,k)))
    return s
    
def get_A(state): # Matrix used by Gros et al 
    up, dn = get_up_dn_sites(state)
    A = np.zeros((int(N/2),int(N/2)))
    for i in range(int(N/2)):
        i_dn = dn[i]
        for j in range(int(N/2)):
            j_up = up[j]
            rij = -get_rij(i_dn,j_up)
            A[i,j] = get_a(rij)
    return A

def get_coeff(state):
    A = get_A(state)
    return np.linalg.det(A)

def get_A_debug(state): # Matrix used by Gros et al 
    up, dn = get_up_dn_sites(state)
    A = np.zeros((int(N/2),int(N/2)),dtype=tuple)
    # print('dn', dn)
    # print('up',up)
    for i in range(int(N/2)):
        i_dn = dn[i]
        for j in range(int(N/2)):
            j_up = up[j]
            # rij = -get_rij(i_dn,j_up)
            # A[i,j] = get_a(rij)
            A[i,j] = (i_dn, j_up)
    return A



# A = get_A(x1)
# print(np.linalg.det(A))

def flip_random(state):
    N = len(state)
    x = np.random.randint(low=0, high=N)
    y = x
    while( state[y] * state[x] > 0):
        y = np.random.randint(low=0, high=N)
    new_state = state.copy()
    new_state[x] *= -1
    new_state[y] *= -1
    return new_state

if 0:
    x1 = make_ising_state2D(L)
    # print(x1+np.ones(N)*0.5)
    # A1 = get_A_debug(x1)
    A1 = get_A(x1)
    
    # x2 = flip_random(x1)
    # x2 = swap_spins(x1, 0, 1)
    x2 = swap_spins(x1, 0, 6)
    A2 = get_A(x2)
    
    A12 = A2-A1
    print(x1+np.ones(N)*0.5)
    
    up, dn = get_up_dn_sites(x1)
    A1d = get_A_debug(x1)
    print('dn', dn)
    print('up',up)
    
    print(x2+np.ones(N)*0.5)
    up, dn = get_up_dn_sites(x2)
    A2d = get_A_debug(x2)
    print('dn', dn)
    print('up',up)

########################################################################################


# def get_A_debug2(state, up, dn): # Matrix used by Gros et al 
#     # up, dn = get_up_dn_sites(state)
#     A = np.zeros((int(N/2),int(N/2)),dtype=tuple)
#     print('dn', dn)
#     print('up',up)
#     for i in range(int(N/2)):
#         i_dn = dn[i]
#         for j in range(int(N/2)):
#             j_up = up[j]
#             # rij = -get_rij(i_dn,j_up)
#             # A[i,j] = get_a(rij)
#             A[i,j] = (i_dn, j_up)
#     return A

def get_A2(state, up, dn): # Matrix used by Gros et al 
    # up, dn = get_up_dn_sites(state)
    A = np.zeros((int(N/2),int(N/2)))
    for i in range(int(N/2)):
        i_dn = dn[i]
        for j in range(int(N/2)):
            j_up = up[j]
            rij = -get_rij(i_dn,j_up)
            A[i,j] = get_a(rij)
    return A


def swap_spins2(state, up, dn, i, j):
    # now i, j refer to the index of the up/dn sites we will flip i, j in (0, ..., N/2)
    dnsite = dn[i]
    upsite = up[j]
    up[j] = dnsite
    dn[i] = upsite
    
    new_state = state.copy()
    si = state[dnsite]
    sj = state[upsite]
    new_state[dnsite] = sj
    new_state[upsite] = si
    return new_state, up, dn


# x3, up, dn = swap_spins2(x1, up, dn, 3, 0)
    
# print(x3+np.ones(N)*0.5)
# print('dn', dn)
# print('up',up)
    
# A3 = get_A2(x2, up ,dn)

# print(np.linalg.det(A2))
# print(np.linalg.det(A3))

if 0:
    x1 = make_ising_state2D(L)
    up, dn = get_up_dn_sites(x1)
    
    print(x1+np.ones(N)*0.5)
    print('dn', dn)
    print('up',up)
    
    A1 = get_A2(x1, up, dn)
    x2, up, dn = swap_spins2(x1, up, dn, 7, 0)
    
    print(x2+np.ones(N)*0.5)
    print('dn', dn)
    print('up',up)
    
    A2 = get_A2(x2, up ,dn)
    
    A12 = A2-A1
    # print('#####################')
    print(np.linalg.det(A1))
    print(np.linalg.det(A2))
    
    
def flip_random2(state, up, dn):
    N = len(state)
    i = np.random.randint(low=0, high=int(N/2))
    j = np.random.randint(low=0, high=int(N/2))
    # print(i,j)
    new_state, up, dn = swap_spins2(state, up, dn, i, j)
    return new_state, up, dn

if 0:
    # checking update procedure where we update up/dn lists
    x1 = make_ising_state2D(L)
    # print(x1)
    # print_formatted(x1)
    up, dn = get_up_dn_sites(x1)
    A1 = get_A2(x1, up, dn)
    print(np.linalg.det(A1))
    
    
    x2, up, dn = flip_random2(x1, up, dn)
    A2 = get_A2(x1, up, dn)
    print(np.linalg.det(A2))
    
    
    A12 = A2-A1
    



if 0:
    for i in range(int(N/2)):
        rij = get_rij(0,i)
        r = np.sqrt(rij[0]**2 + rij[1]**2)
        print('r=',rij, '\t a(r)=', get_a(rij))


#######################################################################################
#


''' TILTED PBC '''
if 0:
    # kk = get_klist(L)
    
    def get_klist_tilted(L):
        kk = []
        for m in range(-2,3):
            for n in range(-2,3):
                kx = 2*pi*(m*L-n)/(L**2 + 1)
                ky = 2*pi*(m+n*L)/(L**2 + 1)
                kvec = np.asarray([kx,ky])
                kk.append(kvec)
        kk.append(np.asarray([pi,pi]))
        return kk
    # def get_klist(L):
    #     kk = []
    #     for nx in range(0,L):
    #         for ny in range(0,L):
    #             kx = 2*pi/L*nx
    #             ky = 2*pi/L*ny + pi/L
    #             kvec = np.asarray([kx,ky])
    #             kk.append(kvec)
    #     return kk
    
    
    kk = get_klist_tilted(5)
    if 1: # if you want to plot the 1st BZ and the k points 
        #   we use in the discrete FT
        
        rect = patches.Rectangle((-1,-1), 2, 2, color='lightcoral')
        pl.figure()
        pl.axes()
        pl.gca().add_patch(rect)
        pl.gca().set(xlim=(-1.4,1.4),ylim=(-1.4,1.4))
        pl.grid()
        for k in kk:
                pl.plot(k[0]/pi, k[1]/pi, 'ko')
        pl.xlabel('$k_x/\pi$',fontsize=14)
        pl.ylabel('$k_y/\pi$',fontsize=14)
        pl.show()
        





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
                c = get_coeff(state_new)/coeff*J1
                if 0: # False (true) for (anti) PBC
                    if i>= L**2-L and i <= L**2-1:
                        c*= -1
                    if i>=0 and i<= L-1:
                        c*= -1
                ssum += c

    return res - 0.5*ssum

def metro(L, Delta, nn, plot, Nsample, Nskip):
    J1 = 1
    Nsite = L**2
    x = get_random_state(L)
    up,dn = get_up_dn_sites(x)
    # print('dn', dn)
    # print('up',up)
    
    EE = np.zeros(Nsample)
    Etot = 0
    
    eL_list = np.zeros(Nsample)
    
    for i in range(Nsample):
        # print(i)
        for j in range(Nskip):
            x1, up1, dn1 = flip_random2(x, up, dn)
            # print('dn', dn)
            # print('up',up)
            A0 = get_A2(x, up, dn)
            A1 = get_A2(x1, up1, dn1)
            
            
            coeff_old = np.linalg.det(A0)
            coeff_new = np.linalg.det(A1)
            print(x+np.ones(N)*0.5)
            print(up, dn)
            print(coeff_new)
            print()
            
            if( np.random.random() < min(1.0, np.abs(coeff_new/coeff_old)**2)):
                x = x1.copy()
                coeff_old = coeff_new
                up = up1.copy()
                dn = dn1.copy()
                
        eL = local_energy(x, coeff_old, Nsite, nn, J1)
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
    # print(EE[-1]/Nsite, sig/Nsite)
    
    return EE[-1]/Nsite, sig/Nsite

nn = get_nn1_array(L)
Delta=5

E, sig = metro(L,Delta,nn,1,10,3)





t1 = time.time()
print("Elapsed time: %.2f sec" % (t1 - t0))