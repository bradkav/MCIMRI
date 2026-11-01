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
RESAMPLE_B          = False
DO_INNER            = True
SIMULTANEOUS        = False
DO_RESONANCES       = False
DO_CAPTURE          = False


def calculate_dE(Es, Ls, Lz, binary, r_orb, mult = 1, include_DF=True, include_3body=True, elements="ELLz", r_min = 1e-9*u.pc):
    
    N = len(Es)
    
    #Calculate some parameters of the binary orbit
    v_orb = binary.v_orb(r_orb)
    Torb = binary.T_orb(r_orb)
    
    m1 = binary.m1
    m2 = binary.m2
    
    #----------------
    b_max = 0.3*r_orb
    #b_max = np.sqrt(m2/m1)*r_orb
    #r_min = 1e-9*u.pc
    
    r_min_stir = r_orb# + b_max
    r_max_stir = r_orb# - b_max
    
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

    
    #BJK - new definition...
    v_rel_x = vorb_x - vx
    v_rel_y = vorb_y - vy
    v_rel_z = vorb_z - vz
    
    #General definition!
    v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)

    #BJK - old good definition
    dx = x0 - x_orb
    dy = y0 - y_orb
    dz = z0 - z_orb
    
    #BJK - new definition?
    dr_dot_vrel = dx*v_rel_x/v_rel + dy*v_rel_y/v_rel + dz*v_rel_z/v_rel
    
    b_x = dx - dr_dot_vrel*v_rel_x/v_rel
    b_y = dy - dr_dot_vrel*v_rel_y/v_rel
    b_z = dz - dr_dot_vrel*v_rel_z/v_rel

    bs0 = np.sqrt(b_x**2 + b_y**2 + b_z**2)

    #b90 = np.ones(N)*u.G_N*m2/((dv)**2)
    b90 = u.G_N*m2/((v_rel)**2)
    bs = 1.0*bs0

    b_min = r_min
    #b_min = r_min*np.sqrt(1 + 2*b90/r_min)
    #print(b_min/r_min)
    
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

    protected = bs < 0
    if (DO_RESONANCES):

        # Check resonance conditions
        #Horseshoes:
        semi_maj = u.G_N*m1/(2*Es)
        ecc_sq   = 1 - 2*Es*Ls_a**2/(u.G_N*m1)**2
        ecc_sq   = np.abs(ecc_sq)
        ecc      = np.sqrt(ecc_sq)
        cosinc   = Lz_a/Ls_a
        inc      = np.arccos(cosinc)
    
        q        = m2/m1
        Rh       = r_orb*(q/3)**(1/3)
        #print(Rh/r_orb)
    
        horseshoes = (np.abs(semi_maj - r_orb) < 4*Rh) & (ecc < q**(1/3)) & (inc < q**(1/3))
        trojans    = (np.abs(semi_maj - r_orb) < Rh) & (ecc < q**0.5) & (cosinc > 0.75)
        delta_res  = q**0.5*ecc**0.5
        
        a_21       = r_orb*(1/2)**(2/3)
        res_21     = (np.abs(semi_maj - a_21) < delta_res*a_21) & (ecc > q**(1/3)) & (ecc < 0.6) & (inc*180/np.pi < 20)
        
        a_12       = r_orb*(2)**(2/3)
        res_12     = (np.abs(semi_maj - a_12) < delta_res*a_12) & (ecc > q**(1/3)) & (ecc < 0.6) & (inc*180/np.pi < 20)

        #print("Horseshoe fraction:", np.sum(horseshoes)/N)
        #print("Trojan fraction:", np.sum(trojans)/N)
        #print("2:1 fraction:", np.sum(res_21)/N)
        #print("1:2 fraction:", np.sum(res_12)/N)

        protected = horseshoes | trojans | res_21 | res_12
        #protected = res_21 #s| res_12

    mask_DF = bs < 0
    if (include_DF):

        mask_DF  = (b_min < bs) & (bs < b_max) & (~protected)
        
        N_DF = int(np.sum(mask_DF > 0))
        
        #_x = np.random.rand(N_DF)
        #_p = 0.5*(1 - Lz_new[mask_DF]/L_circ[mask_DF])
        #scatter = _x < _p
        scatter = 1.0
        
        #p0 = 0.1
        
        
        if (N_DF > 0):
            #BJK!
            #p0 = p_scat
            _x = np.random.rand(N_DF)

            #BJK: THIS NEEDS TO BE CHANGED, I THINK!!!
            scatter = _x < p_scat[mask_DF]
            #scatter = np.random.choice(np.array([0, 1]), size=N_DF, replace=True, p = [1-p0, p0])
            
            if (RESAMPLE_B):
                b_new = b_max*np.sqrt(np.random.rand(N_DF))
            else:
                b_new = bs[mask_DF]

            #dv_para = 2*v_orb*(1 + b_new**2/b90[mask_DF]**2)**(-1) 
            #dv_perp = -2*v_orb*(b_new/b90[mask_DF])*(1 + b_new**2/b90[mask_DF]**2)**(-1) 
                        
            dv_para = 2*v_rel[mask_DF]*(1 + b_new**2/b90[mask_DF]**2)**(-1) 
            dv_perp = -2*v_rel[mask_DF]*(b_new/b90[mask_DF])*(1 + b_new**2/b90[mask_DF]**2)**(-1) 

            #j = b_new/b90[mask_DF]
            #chi = 2*np.arctan(1/j) + 3*(2/j)*(v_rel[mask_DF]/u.c)**2*(1 + 1/j**2)**-1
            
            #dv_para  = v_rel[mask_DF]*(1 - np.cos(chi))
            #dv_perp = -v_rel[mask_DF]*np.sin(chi)
            
            dv_para *= scatter
            dv_perp *= scatter        
            
            dv_x[mask_DF] += dv_para*v_rel_x[mask_DF]/v_rel[mask_DF] + dv_perp*b_x[mask_DF]/bs[mask_DF]
            dv_y[mask_DF] += dv_para*v_rel_y[mask_DF]/v_rel[mask_DF] + dv_perp*b_y[mask_DF]/bs[mask_DF]
            dv_z[mask_DF] += dv_para*v_rel_z[mask_DF]/v_rel[mask_DF] + dv_perp*b_z[mask_DF]/bs[mask_DF]
                    
    if (include_3body):
        
        if (DO_INNER):
            a_DM = u.G_N*m1/(2*Es)
            
            if (SIMULTANEOUS):
                #mask_outer = (rs > r_min_stir) & (~protected)
                mask_inner = (rs < r_max_stir) & (a_DM < r_orb) & (~protected)
            else:
                #mask_outer = (rs > r_min_stir) & (~mask_DF) & (~protected)
                mask_inner = (rs < r_max_stir) & (a_DM < r_orb) & (~mask_DF) & (~protected)

                
            norm2 = (1/np.pi)*(u.G_N*m2*Torb*a_DM[mask_inner]/r_orb**3)*(a_DM[mask_inner]/r_orb)**(3/2)

            xhat, yhat, zhat = utilities.get_random_unitvector(len(norm2))

            dv_x[mask_inner]  += norm2*xhat
            dv_y[mask_inner]  += norm2*yhat
            dv_z[mask_inner]  += norm2*zhat

        #Add shift due to 3 body effects
        if (SIMULTANEOUS):
            mask_3b = (rs > r_min_stir) & (~protected)
        else:
            mask_3b = (rs > r_min_stir) & (~mask_DF) & (~protected)

        r0 = rs[mask_3b]
        v0 = vs[mask_3b]
        th0 = theta0[mask_3b]
        ph0 = phi0[mask_3b]
        thv = theta[mask_3b]
        phv = phi[mask_3b]

        #norm = u.G_N*m2*Torb*r_orb**2/r0**4

        xhat = sin(th0)*cos(ph0)
        yhat = sin(th0)*sin(ph0)
        zhat = cos(th0)
        
        #mu = (xhat*vx[mask_3b] + yhat*vy[mask_3b] + zhat*vz[mask_3b])/vs[mask_3b]

        #Monopole
        #dv_x[mask_3b] += norm2*xhat
        #dv_y[mask_3b] += norm2*yhat
        #dv_z[mask_3b] += norm2*zhat

        #Quadrupole
        norm = u.G_N*m2*Torb*r_orb**2/r0**4
        dv_x[mask_3b] += -norm*(3/4)*(1 - 5*np.cos(th0)**2)*xhat
        dv_y[mask_3b] += -norm*(3/4)*(1 - 5*np.cos(th0)**2)*yhat
        dv_z[mask_3b] += -norm*(3/4)*(3 - 5*np.cos(th0)**2)*zhat
            

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
    
