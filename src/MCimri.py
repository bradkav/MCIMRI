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

import binaries
import orbits
import utilities

#################
##### FLAGS #####
#################
DO_INNER            = True
DO_CAPTURE          = False

def calculate_dE(Es, Ls, Lz, binary, r_orb, mult = 1, include_DF=True, include_3body=True, elements="ELLz", b_min = 0.0, b_max=None):
    
    N = len(Es)
    
    #Calculate some parameters of the binary orbit
    v_orb = binary.v_orb(r_orb)
    Torb = binary.T_orb(r_orb)
    
    m1 = binary.m1
    m2 = binary.m2
    
    #----------------
    if (b_max is None):
        b_max = 0.3*r_orb
    
    r_min_stir = r_orb
    r_max_stir = r_orb
    #----------------
    
    L_circ = u.G_N*m1/np.sqrt(2*Es)
    
    Ls_a = Ls
    Lz_a = Lz
    
    if (elements == "E"):
        Ls_a = L_circ*np.sqrt(np.random.rand(N))
    
    if (elements == "EL"):
        Lz_a =  Ls*(2*np.random.rand(N)-1)
        
    rs, vs, x0, y0, z0, vx, vy, vz, Lx, Ly = orbits.sample_phase_space_coordinates(Es, Ls_a, Lz_a, m1)
    
    #----------------------
    theta0 = np.arccos(z0/rs)
    phi0   = np.arctan2(y0, x0)
    
    theta  = np.arccos(vz/vs)
    phi    = np.arctan2(vy, vx)
    
    phi_orb = 1.0*phi0
    
    x_orb = r_orb*np.cos(phi_orb)
    y_orb = r_orb*np.sin(phi_orb)
    z_orb = 0.0

    vorb_x = v_orb*-np.sin(phi_orb)
    vorb_y = v_orb*np.cos(phi_orb)
    vorb_z = 0.0*phi_orb
    
    dv = np.sqrt( (vx - vorb_x)**2 + (vy - vorb_y)**2 + (vz - vorb_z)**2 )

    v_orb_DM = np.sqrt(u.G_N*m1/rs)
    
    v_circ = (vx*vorb_x + vy*vorb_y + vz*vorb_z)/v_orb
    
    p_scat = np.clip(np.abs(1 - v_circ/v_orb_DM), 0, 1)

    v_rel_x = vorb_x - vx
    v_rel_y = vorb_y - vy
    v_rel_z = vorb_z - vz
    
    v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)

    dx = x0 - x_orb
    dy = y0 - y_orb
    dz = z0 - z_orb
    
    dr_dot_vrel = dx*v_rel_x/v_rel + dy*v_rel_y/v_rel + dz*v_rel_z/v_rel
    
    b_x = dx - dr_dot_vrel*v_rel_x/v_rel
    b_y = dy - dr_dot_vrel*v_rel_y/v_rel
    b_z = dz - dr_dot_vrel*v_rel_z/v_rel

    bs0 = np.sqrt(b_x**2 + b_y**2 + b_z**2)

    b90 = u.G_N*m2/((v_rel)**2)
    bs = 1.0*bs0
    
    dE = 0.0*Es
    dL = 0.0*Es
    
    dv_x    = 0.0*vs
    dv_y    = 0.0*vs
    dv_z    = 0.0*vs
    dL_x    = 0.0*vs
    dL_y    = 0.0*vs
    dL_z    = 0.0*vs
    
    ############
    ## CHECKS ##
    ############
    
    assert np.all(rs > 0)
    assert np.all(vs > 0)

    mask_DF = bs < 0
    if (include_DF):

        mask_DF  = (b_min < bs) & (bs < b_max)
        N_DF = int(np.sum(mask_DF > 0))
        
        if (N_DF > 0):
            _x = np.random.rand(N_DF)
            scatter = _x < p_scat[mask_DF]
            
            #Parallel and perperdicular components of the velocity kick            
            dv_para = 2*v_rel[mask_DF]*(1 + b_new**2/b90[mask_DF]**2)**(-1) 
            dv_perp = -2*v_rel[mask_DF]*(b_new/b90[mask_DF])*(1 + b_new**2/b90[mask_DF]**2)**(-1) 

            #Apply probability to scatter
            dv_para *= scatter
            dv_perp *= scatter        
            
            #Project onto v_rel and b components
            dv_x[mask_DF] += dv_para*v_rel_x[mask_DF]/v_rel[mask_DF] + dv_perp*b_x[mask_DF]/bs[mask_DF]
            dv_y[mask_DF] += dv_para*v_rel_y[mask_DF]/v_rel[mask_DF] + dv_perp*b_y[mask_DF]/bs[mask_DF]
            dv_z[mask_DF] += dv_para*v_rel_z[mask_DF]/v_rel[mask_DF] + dv_perp*b_z[mask_DF]/bs[mask_DF]
                    
    if (include_3body):
        
        if (DO_INNER):
            a_DM = u.G_N*m1/(2*Es)
            
            mask_inner = (rs < r_max_stir) & (a_DM < r_orb) & (~mask_DF)
    
            norm2 = (1/np.pi)*(u.G_N*m2*Torb*a_DM[mask_inner]/r_orb**3)*(a_DM[mask_inner]/r_orb)**(3/2)

            xhat, yhat, zhat = utilities.get_random_unitvector(len(norm2))

            dv_x[mask_inner]  += norm2*xhat
            dv_y[mask_inner]  += norm2*yhat
            dv_z[mask_inner]  += norm2*zhat

        #Add shift due to 3 body effects
        mask_outer = (rs > r_min_stir) & (~mask_DF)

        r0 = rs[mask_outer]
        v0 = vs[mask_outer]
        th0 = theta0[mask_outer]
        ph0 = phi0[mask_outer]

        #norm = u.G_N*m2*Torb*r_orb**2/r0**4

        xhat = sin(th0)*cos(ph0)
        yhat = sin(th0)*sin(ph0)
        zhat = cos(th0)
        
        #Quadrupole
        norm = u.G_N*m2*Torb*r_orb**2/r0**4
        dv_x[mask_outer] += -norm*(3/4)*(1 - 5*np.cos(th0)**2)*xhat
        dv_y[mask_outer] += -norm*(3/4)*(1 - 5*np.cos(th0)**2)*yhat
        dv_z[mask_outer] += -norm*(3/4)*(3 - 5*np.cos(th0)**2)*zhat
            

    if (DO_CAPTURE):
        b_cap = 4*u.G_N*m2/(v_rel*u.c)
        mask_cap  = (bs < b_cap)

        #BJK: A temporary fix!
        dv_x[mask_cap] = 1e10*vx[mask_cap]
        dv_y[mask_cap] = 1e10*vy[mask_cap]
        dv_z[mask_cap] = 1e10*vz[mask_cap]
        
    
    vnew_x = vx + dv_x*mult
    vnew_y = vy + dv_y*mult
    vnew_z = vz + dv_z*mult
    
    vnew_sq = (vnew_x)**2 + (vnew_y)**2 + (vnew_z)**2

    dE += 0.5*(vs**2 - vnew_sq)

    Lnew_x = +(y0*vnew_z - z0*vnew_y)
    Lnew_y = -(x0*vnew_z - z0*vnew_x)
    Lnew_z = +(x0*vnew_y - y0*vnew_x)
    
    dL_z += Lnew_z - Lz_a

    Lnew = np.sqrt(Lnew_x**2 + Lnew_y**2 + Lnew_z**2)

    dL += Lnew - Ls

        
    return dE, dL, dL_z
    
