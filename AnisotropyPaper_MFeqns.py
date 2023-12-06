# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:27:22 2023

@author: alexa
"""
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.patches as patches
from numpy import pi, sin, cos, exp

L = 6
N=L**2
klist = []

for n in range(int(-L/2), int(L/2)):
    for m in range(int(-L/2), int(L/2)):
        k_vec = 2*pi/L*np.asarray([n,m])
        klist.append(k_vec)
if 0:    
    rect = patches.Rectangle((-1,-1), 
                             2, 2, color='r')
    
    pl.axes()
    pl.gca().add_patch(rect)
    pl.gca().set(xlim=(-2.1,2.1),ylim=(-2.1,2.1))
    pl.grid()
    for k in klist:
            pl.plot(k[0]/pi, k[1]/pi, 'ko')
    pl.xlabel('$k_x/\pi$')
    pl.ylabel('$k_y/\pi$')
    pl.show()
    
kk = klist
    
alpha = 1
# L=12
# N=L**2

def get_Ek(lamda, Dx, Dy, k):
    kx = k[0]
    ky = k[1]
    return np.sqrt( lamda**2 - 4*(Dx*sin(kx) + alpha*Dy*sin(ky)) )

def get_eq19(lam, Dx, Dy, alpha):
    s=0
    for k in kk:
        s+= lam/N/get_Ek(lam,Dx,Dy,k)
    return s
    
def get_eq20x(lam,Dx,Dy,alpha):
    s = 0
    for k in kk:
        kx = k[0]
        ky = k[1]
        s+= 1/N*sin(kx)*(Dx*sin(kx)+alpha*Dy*sin(ky))/get_Ek(lam,Dx,Dy,k)
    return s

def get_eq20y(lam,Dx,Dy,alpha):
    s = 0
    for k in kk:
        kx = k[0]
        ky = k[1]
        s+= 1/N*sin(ky)*(Dx*sin(kx)+alpha*Dy*sin(ky))/get_Ek(lam,Dx,Dy,k)
    return s

lam = 2
Dx = 0.5
Dy = 0.5

for k in kk:
    print(get_Ek(lam,Dx,Dy,k))

Dx = 0.58
# Dx = 0.5
Dy = Dx
alpha = 1



if 0: # these plots are for getting a basic idea what is going on when we fix Dx, Dy and vary lamda
    lam_list = np.linspace(1,5,100)
    ee19 = []
    ee20x = []

    for lam in lam_list:
        ee19.append(get_eq19(lam, Dx, Dy, alpha))
        ee20x.append(get_eq20x(lam,Dx,Dy,alpha))
        
    pl.figure()
    pl.plot(lam_list, ee19)
    pl.ylabel('Eqn 19 (want = 2)')
    pl.xlabel(r'$\lambda$')
    pl.title(r'$\Delta_x$ ='+'%.2f'%Dx+r' $\Delta_y$ ='+'%.2f \n'%Dy+r' $\alpha$='+'%.1f'%alpha + ' N=%i'%N )
    pl.hlines(2, lam_list[0], lam_list[-1])
    pl.grid()
    pl.show()

    pl.figure()
    pl.plot(lam_list, ee20x)
    pl.ylabel('Eq 20x')
    pl.xlabel(r'$\lambda$')
    pl.title(r'$\Delta_x$ ='+'%.2f'%Dx+r' $\Delta_y$ ='+'%.2f \n'%Dy+r' $\alpha$='+'%.1f'%alpha + ' N=%i'%N )
    pl.hlines(Dx, lam_list[0], lam_list[-1])
    pl.grid()
    pl.show()

lam = 2



def get_eq19(lam, Dx, Dy, alpha):
    s=0
    for k in kk:
        s+= lam/N/get_Ek(lam,Dx,Dy,k)
    return s



tol = 0.0001
def get_lam(lam, Dx,Dy, alpha, tol,plot):
    dl = 0.005
    x = 2 - get_eq19(lam, Dx, Dy, alpha)
    while x > 0:
        #print(x , lam)
        lam -= dl
        x = 2 - get_eq19(lam, Dx, Dy, alpha)
    A = lam # A
    B = lam + dl # B
    C = (A + B)/2
    fc = 2 - get_eq19(C, Dx, Dy, alpha)
    fc_list = []
    lams = []
    while abs(fc) > tol:
        fc_list.append(fc)
        lams.append(C)
        if fc > 0:
            A = A
            B = C 
        elif fc < 0:
            A = C
            B = B
        C = (A+B)/2
        fc = 2-get_eq19(C, Dx, Dy, alpha)
        #print('lam = %.4f, \t y = %.4f'%(C, fc))
    it = np.linspace(0,len(fc_list), len(fc_list))
    if plot==True:
        fig, ax = pl.subplots(2,1, sharex=True)
        ax[0].plot(it, fc_list)
        ax[0].set_ylabel('2 - Eq.19')
        ax[1].plot(it, lams)
        ax[1].set_ylabel(r'$\lambda$')
        ax[1].set_xlabel('iteration')
        ax[0].grid(True)
        ax[1].grid(True)
        ax[0].set_title(r'$\Delta_x=%.3f, \Delta_y=%.3f, \alpha=%.3f$'%(Dx,Dy,alpha))

    return C
        
lam = get_lam(5,Dx,Dy,alpha, 0.000001, 1)
print('lam = ', lam)

print('eq19-2: (should be zero)')
print(get_eq19(lam,Dx,Dy,alpha)-2)


print(get_eq20x(lam,Dx,Dy,alpha))

if 0:
    DD = np.linspace(0.5, 0.6, 100)
    eeq20 = []
    
    for D in DD:
        eeq20.append(D- get_eq20x(lam,D,D,alpha=1) )
    
    pl.figure()
    pl.grid()
    pl.plot(DD,eeq20)
    pl.ylabel('Dx - eq20x(Dx)')
    pl.xlabel(r'$\Delta_x$')
    pl.show()

print()

# def get_Dx(lam, Dx,Dy, alpha, tol,plot):
#     dx = 0.001
#     x = 2 - get_eq20x(lam, Dx, Dy, alpha)
#     while x > 0:
#         # print(x , Dx)
#         Dx += dx
#         x = Dx - get_eq20x(lam, Dx, Dx, alpha)
        
#     # print(x , Dx)
#     # print()
#     A = Dx - dx # A
#     B = Dx # B
#     C = (A + B)/2
#     fc = C - get_eq20x(lam, C, C, alpha)
#     fc_list = []
#     DD = []
#     # print('fc = ', fc)
#     # print('f(A) = ', A - get_eq20x(lam, A, A, alpha))
#     # print('fc = ', fc)
#     # print('f(B) = ', B - get_eq20x(lam, B, B, alpha))
#     while abs(fc) > tol:
#         # print(fc)
#         # print('fc = ', fc)
#         fc_list.append(fc)
#         DD.append(C)
#         if fc < 0:
#             A = A
#             B = C 
#         elif fc > 0:
#             A = C
#             B = B
#         C = (A+B)/2
#         fc = C - get_eq20x(lam, C, C, alpha)
#         # print(C)
#         #print('lam = %.4f, \t y = %.4f'%(C, fc))
#     it = np.linspace(0,len(fc_list), len(fc_list))
#     if plot==True:
#         fig, ax = pl.subplots(2,1, sharex=True)
#         ax[0].plot(it, fc_list)
#         ax[0].set_ylabel('Dx - Eq.20x')
#         ax[1].plot(it, DD)
#         ax[1].set_ylabel(r'$\Delta_x$')
#         ax[1].set_xlabel('iteration')
#         ax[0].grid(True)
#         ax[1].grid(True)
#         ax[0].set_title(r'$\Delta_x=%.3f, \Delta_y=%.3f, \alpha=%.3f$'%(Dx,Dy,alpha))

#     return C


# Dx = get_Dx(lam, 0.5, 0.5, alpha,tol,plot=0)


print()
print(2-get_eq19(lam,Dx,Dx,alpha))
print(Dx-get_eq20x(lam,Dx,Dx,alpha))
print('lam = ',lam)
print('Dx = ', Dx)


# print()
# lam = get_lam(5,Dx,Dx,alpha, 0.000001, 1)
# Dx = get_Dx(lam, 0.5, 0.5, alpha,tol,plot=0)

# print(2-get_eq19(lam,Dx,Dx,alpha))
# print(Dx-get_eq20x(lam,Dx,Dx,alpha))
# print('lam = ',lam)
# print('Dx = ', Dx)
#####################################################################################
import time





def get_Dxy(lam,Dx,Dy,alpha, eps, tol):
    t0 = time.time()
    lam = get_lam(lam,Dx,Dy,alpha, 0.000001, 0)
    dx = get_eq20x(lam,Dx,Dy,alpha)
    dy = get_eq20y(lam,Dx,Dy,alpha)
    Dx_new = Dx*(1-eps) + eps*dx
    Dy_new = Dy*(1-eps) + eps*dy
    lam = get_lam(5,Dx_new,Dy_new,alpha, 0.000001, 0)
    
    while abs(Dx - Dx_new) > tol or abs(Dy - Dy_new) > tol:
        Dx = Dx_new
        Dy = Dy_new
        dx = get_eq20x(lam,Dx,Dy,alpha)
        dy = get_eq20y(lam,Dx,Dy,alpha)
        Dx_new = Dx*(1-eps) + eps*dx
        Dy_new = Dy*(1-eps) + eps*dy
        lam = get_lam(5,Dx_new,Dy_new,alpha, 0.0001, 0)
        print('Dx=',Dx)
    
    lam = get_lam(5,Dx_new,Dy_new,alpha, 0.0001, 0)
    #print(dx)
    t = time.time()
    print('runtime: ', t-t0)
    return Dx_new, Dy_new, lam


# print(get_Dxy(lam,Dx,Dy,alpha,0.2,1e-4))


# do this to solve:
# Dx,Dy,lam = get_Dxy(lam,Dx,Dy,alpha,0.2,1e-4)

# print(2-get_eq19(lam,Dx,Dx,alpha))
# print(Dx-get_eq20x(lam,Dx,Dx,alpha))