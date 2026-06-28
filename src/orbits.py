import numpy as np
from numpy import cos, sin

from tqdm import tqdm

from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from NbodyIMRI import distributionfunctions as df
from NbodyIMRI import units as u
from NbodyIMRI import tools

import utilities

import binaries

#Calculate the gravitational potential of a point mass
def psi(r, M):
    return u.G_N*M/r
    
#https://ui.adsabs.harvard.edu/abs/1987CeMec..40..329M/abstract
def calc_E(M, e, N_iter=1):
    alpha = (1 - e)/(4*e + 0.5)
    beta = 0.5*M/(4*e + 0.5)
    sign_beta = np.where(beta >= 0, 1, -1)
    arg = beta + sign_beta*np.sqrt(beta**2 + alpha**3)
    z = (arg)**(1/3)
    #z = (beta + np.sign(beta)*np.sqrt(beta**2 + alpha**3))**(1/3)
    s = z - alpha/z
    ds = 0.078*s**5/(1+e)
    s += ds
    E_old = M + e*(3*s - 4*s**3)
        
    for i in range(N_iter):
        E_new = E_old - (E_old - e*np.sin(E_old) - M)/(1 - e*np.cos(E_old))
        E_old = 1.0*E_new
        
    return E_old
    
def sample_radii(E, L, m1):
    N = len(E)
    M = 2*np.pi*np.random.rand(N)
    
    e = np.sqrt(1 - 2*E*L**2/(u.G_N*m1)**2)
    if np.any((e < 0) | (e > 1)):
        print(e, E, L)
        assert 1 == 0
    
    Eanom = calc_E(M, e)
    
    a = u.G_N*m1/(2*E)
    r = a*(1 - e*np.cos(Eanom))
    #print(r)
    return r
    

    
def draw_biased_E(DF, r_min, r_max, r0 = -1, N = 1):
    E_min = DF.Psi(r_max)
    E_max = DF.Psi(r_min)
    
    if (r0 < 0):
        E0 = E_max
    else:
        E0 = DF.Psi(r0)
    
    beta = 0.5
    #C = 1000
    C = 0
    A = C/E0**beta
    
    
    integ = lambda E: DF.density_of_states_E(E)*DF.f_ini(E)*(1 + A*E**beta)
    E = tools.inverse_transform_sample(integ, E_min, E_max, N, N_grid=10000, log=True)
    w = 1/(1 + A*E**beta)
    return E, w
    
def sample_phase_space_coordinates(Es, Ls, Lz, m1):
    
    N = len(Es)
    
    rs = sample_radii(Es, Ls, m1)
    
    #Calculate the corresponding velocity
    vs = np.sqrt(2*(psi(rs, m1) - Es))
    
    thetaL = np.arccos(Lz/Ls)
    phiL   = 2*np.pi*np.random.rand(N)
    
    Lhat_x   = sin(thetaL)*cos(phiL)
    Lhat_y   = sin(thetaL)*sin(phiL)
    Lhat_z   = cos(thetaL)
    
    e1x, e1y, e1z = utilities.full_cross(0, 0, 1, Lhat_x, Lhat_y, Lhat_z, normed=True)
    e2x, e2y, e2z = utilities.full_cross(Lhat_x, Lhat_y, Lhat_z, e1x, e1y, e1z)
        
    phase = 2*np.pi*np.random.rand(N)
    
    rhatx = cos(phase)*e1x + sin(phase)*e2x
    rhaty = cos(phase)*e1y + sin(phase)*e2y
    rhatz = cos(phase)*e1z + sin(phase)*e2z
    
    x0 = rs*rhatx
    y0 = rs*rhaty
    z0 = rs*rhatz
    
    inward = np.random.choice(np.array([-1, 1]), size=N, replace=True)
    
    vt = Ls/rs
    vr = inward*np.sqrt(vs**2 - vt**2)
    
    vt_x, vt_y, vt_z = utilities.full_cross(Lhat_x, Lhat_y, Lhat_z, rhatx, rhaty, rhatz)
    
    vx = vr*rhatx + vt*vt_x
    vy = vr*rhaty + vt*vt_y
    vz = vr*rhatz + vt*vt_z
    
    Lx = Ls*Lhat_x
    Ly = Ls*Lhat_y
    
    return rs, vs, x0, y0, z0, vx, vy, vz, Lx, Ly
    
#Reconstruct the density as a function of r, for a given sample of eneries Es
#This is NOT normalised (you have to also specify the mass contributed by each particle to get the correct mass of the spike)
def reconstruct_density(r, Es, m1):
    rho = 0.0*r
    
    for E in Es:
        if (E > 0):
            #Assuming the initial samples are drawn from P(E) = g(E)*f(E)
            rho += weights[i]*np.sqrt( 2*(np.clip(psi(r, m1) - E, 0, 1e30)) )/SpikeDF.density_of_states_E(E)

    return rho


def reconstruct_density_full(r, Es, Ls, weights, m1):
    rho = 0.0*r
    
    N_out = 0
    for i, E in enumerate(Es):
        if (E > 0):
            L_circ = u.G_N*m1/np.sqrt(2*E)
            if (Ls[i] <= L_circ):
                
                vr_sq = 2*(psi(r, m1) - E) - Ls[i]**2/r**2
                inds = vr_sq > 0
                vr = np.sqrt(np.clip(vr_sq, 0, 1e30))

                T = 2*np.pi*u.G_N*m1/((2*E)**1.5)

                #Assuming the initial samples are drawn from P(E) = g(E)*f(E)
                #w = 1/SpikeDF.density_of_states_E(E)
                #w = 1/(Ls[i]*T)
                w = weights[i]
                rho[inds] += (w/(4*np.pi*r[inds]**2))*(1/T)*(2/vr[inds])
                
            else:
                N_out += 1
    #print(N_out)
    return rho
    
def reconstruct_density_full_th(r, theta, Es, Ls, Lz, weights, m1):
    rho = 0.0*theta
    
    N_out = 0
    for i, E in enumerate(Es):
        if (E > 0):
            L_circ = u.G_N*m1/np.sqrt(2*E)
            if (Ls[i] <= L_circ):
                
                vr_sq = 2*(psi(r, m1) - E) - Ls[i]**2/r**2
                if (vr_sq < 0):
                    continue
                
                eps = (1e-10*Ls[i])**2
                
                
                Lsq   = Ls[i]**2*np.sin(theta)**2 - Lz[i]**2
                
                inds = (Lsq > 0)
                vr = np.sqrt(np.clip(vr_sq, 0, 1e30))
                #Lt = np.sqrt(np.clip(Lsq, 0, 1e30))
                
                
                Lt = np.sqrt(np.maximum(Lsq, eps))

                T = 2*np.pi*u.G_N*m1/((2*E)**1.5)

                #Assuming the initial samples are drawn from P(E) = g(E)*f(E)
                #g = E**(-5/2)
                #w = 1/(Ls[i]*T)
                w = weights[i]
                rho[inds] += (w/(4*np.pi*r**2))*(2/vr)*(1/T)*(2*Ls[i]/Lt[inds])
                #rho[inds] += (2*Ls[i]/Lt[inds])
                
            else:
                N_out += 1
    #print(N_out)
    return rho