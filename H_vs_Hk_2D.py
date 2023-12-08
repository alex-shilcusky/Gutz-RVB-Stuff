import numpy as np 
import cmath
import time
from matplotlib import pyplot as pl
import matplotlib.patches as patches
from numpy import sin, cos, pi, linalg, sqrt, exp
import itertools



L = 4
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
    H = np.zeros(shape=(2*N,2*N),dtype=complex)
    for i in range(N):
        H[i,nn[i,0]] = -t 
        H[i,nn[i,1]] = -t
        H[i,nn[i,2]] = -t 
        H[i,nn[i,3]] = -t
        H[i+N,nn[i,0]+N] = t
        H[i+N, nn[i,1]+N] = t
        H[i+N,nn[i,2]+N] = t
        H[i+N,nn[i,3]+N] = t
        
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

nn = get_nn1_array(L)
H = get_H1a(N,t,Delta,nn)

(w,U) = np.linalg.eigh(H)

#H(r) = cdag H c = cdag (U D Udag) c
########################################################################################

def get_klist(L):
    kk = []
    for nx in range(0,L):
    # for nx in range(int(-L/2), int(L/2)):
        for ny in range(0,L):
        # for ny in range(int(-L/2), int(L/2)):
            kx = 2*pi/L*nx
            ky = 2*pi/L*ny + pi/L
            # ky = 2*pi/L*ny
            kvec = np.asarray([kx,ky])
            kk.append(kvec)
    return kk

kk = get_klist(L)
       
if 0: # if you want to plot the 1st BZ and the k points 
    #   we use in the discrete FT
    rect = patches.Rectangle((-1,-1), 
                             2, 2, color='r')
    
    pl.figure()
    pl.axes()
    pl.gca().add_patch(rect)
    pl.gca().set(xlim=(-2.1,2.1),ylim=(-2.1,2.1))
    pl.grid()
    for k in kk:
            pl.plot(k[0]/pi, k[1]/pi, 'ko')
    pl.xlabel('$k_x/\pi$',fontsize=14)
    pl.ylabel('$k_y/\pi$',fontsize=14)
    pl.show()
        
def get_Deltak(Delta, k):
    kx = k[0]
    ky = k[1]
    return 2*Delta*(cos(kx) - cos(ky))
    # return Delta


def get_xik(k):
    kx = k[0]
    ky = k[1]
    return -2*t*(cos(kx) + cos(ky))

def get_Hk(L,kk):
    N = L**2
    H = np.zeros(shape=(2*N,2*N),dtype=complex)
    # i_k = 0
    for i in range(N):
        # k = kk[i_k]
        k = kk[i]
        H[i,i] = get_xik(k)
        H[i+N,i+N] = -get_xik(k)
        H[i+N,i] = np.conj(get_Deltak(Delta,k))
        H[i, i+N] = get_Deltak(Delta,k)
    return H

Hk = get_Hk(L,kk)
(wk,Uk) = np.linalg.eigh(Hk)

'Check that H eigenvalues are same as Hk'
# dw = wk-w
# print(np.linalg.norm(dw))

'Check that indeed Hk = Uk^dag D Uk'
Ukdag = np.conj(Uk.T)

Udag = np.conj(U.T)
Dr = Udag@H@U
Dk = Ukdag@Hk@Uk


# print(np.linalg.norm(Dk-Dr))


'''Let us now make a matrix T which transform k-space creation basis into real
 space basis
(c_1up,...,c_Nup,c_1dn^dag,...,c_Ndn^dag) =
 T(c_k1up,...,c_kNup,c_-k1dn^dag,...,c_-kNdn^dag)
'''

T = np.zeros((2*N,2*N),dtype=complex)
for i in range(N): # rows are determined by position vectors
    # print(site_indices[i])
    ri = site_indices[i]
    for j in range(N): # columns are k-vectors
        kj = kk[j]
        T[i,j] = np.exp(1j*np.inner(kj,ri))/np.sqrt(N)
        T[i+N,j+N] = np.exp(1j*np.inner(kj,ri))/np.sqrt(N)
       
    
Tdag = np.conj(np.transpose(T))
# Id = Tdag@T


' Check that Hr = T Hk Tdag'
H_r = T @ Hk @ Tdag  # use underscore _r or _k to indicate calculation by F.T.

# print(np.linalg.norm(H-H_r))

'YES! The linear transfo T is indeed working!!!'

'Now check that Hk = Tdag Hr T '

H_k = Tdag @ H @ T



# print(np.linalg.norm(Tdag@U - Uk)) # this not being zero is a problem!!!
# print(np.linalg.norm(Udag@T - Ukdag))


# for i in range(2*N):
#     for j in range(2*N):
#         if abs(H_k[i,j]) < 1e-12:
#             H_k[i,j] = 0
            
# for i in range(2*N):
#     for j in range(2*N):
#         if abs(Hk[i,j]) < 1e-12:
#             Hk[i,j] = 0

H_k = np.around(H_k, 7)
Hk = np.around(Hk, 7)

(wk,Uk) = np.linalg.eigh(Hk)
# (w_k, U_k) = np.linalg.eigh(H_k)

print('norm(Hk- Tdag H T): \t ' ,np.linalg.norm(Hk-H_k))


(w,U) = np.linalg.eigh(H)
# print(w)



# U_k = Tdag@U

# Uk = np.around(Uk,7)
# U_k = np.around(U_k, 7)

# # print(np.linalg.norm(Uk-U_k))
# print(np.linalg.det(Uk))
# print(np.linalg.det(U_k))

# # U_r = np.around(T@Uk, 7)
# U_r = T@Uk


# D_r = np.conj(U_r.T) @ H @ U_r
# # D_r = np.around(D_r, 7)


# H_r = U_r @ Dr @ np.conj(U_r.T) 
# H_r = np.around(H_r, 7)

# Diff = Hk - H_k

# for i in range(2*N):
#     for j in range(2*N):
#         if abs(Diff[i,j]) < 1e-12:
#             Diff[i,j] = 0
            
# for i in range(2*N):
#     for j in range(2*N):
#         if abs(Diff[i,j]) < 1e-12:
#             Diff[i,j] = 0

# Diff = np.around(Diff, 3)
# print(np.linalg.det(Hk))
# print(np.linalg.det(H_k))
# print(np.linalg.det(H))
# print(np.linalg.det(H)-np.linalg.det(Hk))
# print(np.linalg.norm(Hk-H_k))
# print()
# print(np.linalg.norm(Uk-U_k))
# Diff = Uk - U_k
# print(np.linalg.det(Uk))
# print(np.linalg.det(U_k))

# print()



# phi = pi/2


def get_E(k):
    xik = get_xik(k)
    Dk = get_Deltak(Delta, k)
    return np.sqrt(xik**2 + abs(Dk)**2)

# for k in kk:
#     print(get_E(k))

def get_uk(k):
    xik = get_xik(k)
    Ek = get_E(k)
    return np.sqrt(1/2*(1+xik/Ek))
    # return np.sqrt(1/2*(1+xik/Ek))*exp(-1j*phi/2)

def get_vk(k):
    xik = get_xik(k)
    Ek = get_E(k)
    return np.sqrt(1/2*(1-xik/Ek))
    # return np.sqrt(1/2*(1-xik/Ek))*exp(1j*phi/2)
    
if 0:
    for k in kk:
        print('uk = %.5f \t vk = %.5f'%(get_uk(k),get_vk(k)))

# Uk1dag = np.zeros((2*N,2*N),dtype=complex)
# for i in range(N):
#     ki = kk[i]
#     Uk1dag[i,i] = get_uk(ki)
#     Uk1dag[i+N,i+N] = np.conj(get_uk(ki))
#     Uk1dag[i,i+N] = get_vk(ki)
#     Uk1dag[i+N,i] = -np.conj(get_vk(ki))

# Uk1 = np.conj(np.transpose(Uk1dag))

# Id = Uk1dag@Uk1
# D = Uk1dag @ Hk @ Uk1
# for k in kk:
#     uk = get_uk(k)
#     vk = get_vk(k)
#     xik = get_xik(k)
#     Dk = get_Deltak(Delta, k)
#     print(get_E(k))

    # print(-2*xik*vk*uk + Dk*uk**2 - Dk*vk**2)
    
    # print(get_E(k), '\t', uk**2 + vk**2)
# print(Uk1dag@Uk1)

# for i in range(2*N):
    # print(wk[i], '\t ', w_k[i])
    # print(wk[i] - w_k[i])
    
if 0:
    UUk = Uk.copy()
    for i in np.argsort(wk):
        UUk[:,i] = Uk[:,i]
        
    UU_k = Uk.copy()
    for i in np.argsort(w_k):
        UU_k[:,i] = U_k[:,i]
    
    # print(Hk@UUk[:,0]-wk[0]*UUk[:,0])
    print(np.linalg.norm(UUk-UU_k))
    # print(np.linalg.norm(wk-w_k))
    
    
    print(np.linalg.norm(Uk-U_k))
########################### 





# H_k = Tdag@H@T



# print(np.linalg.norm( Tdag@U - Uk ))

# print(np.linalg.norm(Hk-H_k))

# H_k_diff = Hk-H_k
# print(np.linalg.norm(H_k_diff))
# (w_k, U_k) = np.linalg.eigh(H_k) 

# print(np.linalg.norm(Uk-U_k))

###############################################




# print(np.linalg.norm( Tdag@U - Uk  ))

# D = np.zeros((2*N,2*N))
# for i in range(2*N):
#     D[i,i] = wk[i]
    
# RHS = Uk@D@Ukdag
# dHk = Hk-RHS

# print(np.linalg.norm(dHk))

# D = 

# #######################################



def get_ak(k):
    Delta_k = get_Deltak(Delta,k)
    xi_k = get_xik(k)
    return Delta_k/(xi_k + np.sqrt( xi_k**2 + Delta_k**2 ))

# def get_uk_vk(k):
#     tanx = get_tanx(k)
#     x = np.arctan(tanx)
#     return np.cos(x), np.sin(x)

# def get_tanx(k):
#     Dk = get_Deltak(Delta,k)
#     xik = get_xik(k)
#     tanx = Dk/(xik + np.sqrt(xik**2 + abs(Dk)*22))
#     return tanx



# for k in kk:
#     uk = get_uk(k)
#     vk = get_vk(k)
#     print(uk**2 + vk**2)
    
    
# for k in kk:
#     # uk, vk = get_uk_vk(k)
#     uk = get_uk(k)
#     vk = get_vk(k)
#     # print( 'uk^2+vk^2', uk**2 + vk**2)
#     ak = get_ak(k)
#     print(vk/uk,'\t',ak)
#     # print(uk**2 + vk**2)
#     xik = get_xik(k)
#     Dk = get_Deltak(Delta, k)
#     print(-2*xik*vk*uk + Dk*uk**2-Dk*vk**2)
    
# for k in kk:
#     xik = get_xik(k)
#     Dk = get_Deltak(Delta,k)
#     # print('Ek = ', np.sqrt(xik**2 + abs(Dk)**2))
#     print('xi_k = ', xik)
    # uk = get_uk(k)
    # vk = get_vk(k)
    # print('uk = ', uk, '\t vk = ', vk, '\t Ek = ', np.sqrt(xik**2 + abs(Dk)**2), '\t uk^2+vk^2', uk**2 + vk**2)
    # print('uk = ', uk, '\t vk = ', vk, '\t Ek = ', np.sqrt(xik**2 + abs(Dk)**2), '\t uk^2+vk^2', uk**2 + vk**2)
    # print('tanx_k = ', get_tanx(k), np.arctan(get_tanx(k)))
    # print('vk = ', vk)


# print(get_coeff1(Uk,x))
# v = Uk[:,0]

# print(Hk@v - wk[0]*v)




# def get_up_dn_sites(state): # function to get lists of up and down sites
#         Nsite = len(state)
#         sites_up = np.zeros(int(Nsite/2),dtype=int)
#         sites_dn = np.zeros(int(Nsite/2),dtype=int)
#         i=0
#         j=0
#         n=0
#         while n < Nsite:
#             if state[n] < 0:
#                 sites_dn[i] = n
#                 i+=1
#             if state[n] > 0:
#                 sites_up[j] = n
#                 j+=1
#             n+=1 
#         return sites_up, sites_dn
    
# def get_coeff1(U,state):
#     N = len(state)
#     Nsite = len(state)
#     up, dn = get_up_dn_sites(state)
#     Ne=len(up)
#     A = np.zeros((Nsite,Nsite),dtype=complex)
#     for i in range(Nsite):
#         for j in range(Ne):
#             A[j,i] = U[up[j],i]
#             A[j+Ne,i] = U[up[j]+N,i]
#     return np.linalg.det(A)
# def make_ising_state2D(L): # function to make an Ising state of size L 
#     N = L**2
#     state = []
#     sgn = 1
#     for i in range(N):
#         if i%L ==0:
#             sgn *= -1
#         state.append(1*sgn)
#         sgn *= -1
#     return np.asarray(state)*0.5 # returns as [0.5, -0.5, ...]

# x = make_ising_state2D(L)
# print(get_coeff1(U,x))

