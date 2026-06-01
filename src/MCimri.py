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

#################
##### FLAGS #####
#################
RESAMPLE_B = False
DO_INNER   = True
SIMULTANEOUS = False
DO_RESONANCES = False


def calculate_dE(Es, Ls, Lz, binary, r_orb, mult = 1, include_DF=True, include_3body=True, include_resonance=True, elements="ELLz"):
    
    N = len(Es)
    
    #Calculate some parameters of the binary orbit
    v_orb = binary.v_orb(r_orb)
    Torb = binary.T_orb(r_orb)
    
    m1 = binary.m1
    m2 = binary.m2
    
    #----------------
    b_max = 0.3*r_orb
    r_min = 1e-9*u.pc
    
    r_min_stir = r_orb# + b_max
    r_max_stir = r_orb# - b_max
    
    #----------------
    
    L_circ = u.G_N*m1/np.sqrt(2*Es)
    
    Ls_new = Ls
    Lz_new = Lz
    
    if (elements == "E"):
        Ls_new = L_circ*np.sqrt(np.random.rand(N))
    
    if (elements == "EL"):
        Lz_new =  Ls*(2*np.random.rand(N)-1)
        
    rs, vs, x0, y0, z0, vx, vy, vz, Lx, Ly = orbits.sample_phase_space_coordinates(Es, Ls_new, Lz_new, m1)
    
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
    
    dx = x0 - x_orb
    dy = y0 - y_orb
    dz = z0 - z_orb
    
    v_circ = (vx*vorb_x + vy*vorb_y + vz*vorb_z)/v_orb
    v_circ2 = (vx*vorb_x + vy*vorb_y + vz*vorb_z)/vs
    
    #v_rel = -v_circ2
    #v_rel = np.sign(v_circ)*np.sqrt((vorb_x - vx)**2 + (vorb_y - vy)**2 + (vorb_z - vz)**2)
    
    #vr_x = v_orb - vx
    #vr_y = v_orb - vy
    #vr_z = v_orb - vz
    
    v_rel = (v_orb - v_circ)
    #v_rel = np.ones(N)*v_orb

    #BJK: Perhaps apply a scattering probability which goes like v_t/v_circ... the kicks shouldn't depend on v_rel...

    bs0 = np.sqrt((x0 - x_orb)**2 + (y0 - y_orb)**2 + (z0 - z_orb)**2)
    
    b90  = np.ones(N)*u.G_N*m2/((v_orb)**2)
    #b90 = u.G_N*m2/((v_rel)**2)
    bs = 1.0*bs0

    b_min = r_min*np.sqrt(1 + 2*b90/r_min)
    
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
        ecc_sq   = 1 - 2*Es*Ls_new**2/(u.G_N*m1)**2
        ecc_sq   = np.abs(ecc_sq)
        ecc      = np.sqrt(ecc_sq)
        cosinc   = Lz_new/Ls_new
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

        #protected = horseshoes | trojans | res_21 | res_12
        protected = res_21 #s| res_12

    mask_DF = bs < 0
    if (include_DF):

        mask_DF  = (b_min < bs) & (bs < b_max) & (~protected)
        
        N_DF = int(np.sum(mask_DF > 0))
        
        #_x = np.random.rand(N_DF)
        #_p = 0.5*(1 - Lz_new[mask_DF]/L_circ[mask_DF])
        #scatter = _x < _p
        scatter = 1.0
        
        #p0 = 0.1
        #scatter = np.random.choice(np.array([0, 1]), size=N_DF, replace=True, p = [1-p0, p0])
        
        if (N_DF > 0):
            
            if (RESAMPLE_B):
                b_new = b_max*np.sqrt(np.random.rand(N_DF))
            else:
                b_new = bs[mask_DF]
                        
            dv_para = 2*v_rel[mask_DF]*(1 + b_new**2/b90[mask_DF]**2)**(-1) 
            dv_perp = -2*v_rel[mask_DF]*(b_new/b90[mask_DF])*(1 + b_new**2/b90[mask_DF]**2)**(-1) 
            
            dv_para *= scatter
            dv_perp *= scatter
            
            dv_x[mask_DF] += dv_para*vorb_x[mask_DF]/v_orb + dv_perp*dx[mask_DF]/bs[mask_DF]
            dv_y[mask_DF] += dv_para*vorb_y[mask_DF]/v_orb + dv_perp*dy[mask_DF]/bs[mask_DF]
            dv_z[mask_DF] += dv_para*vorb_z[mask_DF]/v_orb + dv_perp*dz[mask_DF]/bs[mask_DF]
                    
    if (include_3body):
        
        if (DO_INNER):
            if (SIMULTANEOUS):
                #mask_outer = (rs > r_min_stir) & (~protected)
                mask_inner = (rs < r_max_stir) & (~protected)
            else:
                #mask_outer = (rs > r_min_stir) & (~mask_DF) & (~protected)
                mask_inner = (rs < r_max_stir) & (~mask_DF) & (~protected)
        
            #r0 = rs[mask_outer]
            #th0 = theta0[mask_outer]
            #ph0 = phi0[mask_outer]
            #
            #norm = -u.G_N*m2*Torb/r0**2
            #
            #xhat = sin(th0)*cos(ph0)
            #yhat = sin(th0)*sin(ph0)
            #zhat = cos(th0)
            #
            #dv_x[mask_outer] += norm*(-3/8)*(r/r0)**2*(3 + 5*cos(2*th0))*xhat
            #dv_y[mask_outer] += norm*(-3/8)*(r/r0)**2*(3 + 5*cos(2*th0))*yhat
            #dv_z[mask_outer] += norm*(+3/4)*(r/r0)**2*(-2 + 5*sin(th0)**2)*zhat
            
            r0 = rs[mask_inner]
            th0 = theta0[mask_inner]
            ph0 = phi0[mask_inner]
            
            norm2 = u.G_N*m2*Torb/r_orb**2

            xhat = sin(th0)*cos(ph0)
            yhat = sin(th0)*sin(ph0)
            zhat = cos(th0)
            
            dv_x[mask_inner] += -norm2*(1/32)*(r0/r_orb)**3*(27 + 45*cos(2*th0))*xhat
            dv_y[mask_inner] += -norm2*(1/32)*(r0/r_orb)**3*(27 + 45*cos(2*th0))*yhat
            dv_z[mask_inner] += norm2*(1/8)*(r0/r_orb)**3*(-3 + 15*cos(2*th0))*zhat
            
            
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

        norm = u.G_N*m2*Torb*r_orb**2/r0**4

        xhat = sin(th0)*cos(ph0)
        yhat = sin(th0)*sin(ph0)
        zhat = cos(th0)
        
        mu = (xhat*vx[mask_3b] + yhat*vy[mask_3b] + zhat*vz[mask_3b])/vs[mask_3b]

        #Monopole
        #dv_x[mask_3b] += norm2*xhat
        #dv_y[mask_3b] += norm2*yhat
        #dv_z[mask_3b] += norm2*zhat

        #Quadrupole
        norm = u.G_N*m2*Torb*r_orb**2/r0**4
        dv_x[mask_3b] += norm*(3/8)*(3 + 5*cos(2*th0))*xhat
        dv_y[mask_3b] += norm*(3/8)*(3 + 5*cos(2*th0))*yhat
        dv_z[mask_3b] += norm*(-3/4)*(-2 + 5*sin(th0)**2)*zhat

        #Quadrupole velocity term
        #H = np.clip(v0*Torb/(2*np.pi*r0), 0, 0.1)
        #H = v0*Torb/(2*np.pi*r0)
        #print(np.sum(H > 1))
        #dv_x[mask_3b] += norm*3*sin(th0)*(H/8) * ((3 + 5*cos(2*th0))*(-4*np.pi*mu*cos(ph0) + mu*sin(ph0)) - 10*mu*sin(th0)**2*sin(3*ph0))
        #dv_y[mask_3b] += norm*3*(H/16) * (mu*(20*cos(3*ph0)*sin(th0)**3 + cos(ph0)*(sin(th0) + 5*sin(3*th0))) - 4*np.pi*mu*(sin(th0) + 5*sin(3*th0))*sin(ph0))
        #dv_z[mask_3b] += norm*3*cos(th0)*(H/4) * (-2*np.pi*mu*(-1 + 5*cos(2*th0)) - 10*mu*sin(th0)**2*sin(2*ph0))

        sθ, cθ = np.sin(th0), np.cos(th0)
        sφ, cφ = np.sin(ph0), np.cos(ph0)

        sθv, cθv = np.sin(thv), np.cos(thv)
        sφv, cφv = np.sin(phv), np.cos(phv)

        s3θ = np.sin(3*th0)
        
        #H = np.clip(v0*Torb/(2*np.pi*r0), 0, 1.0)
        H = v0*Torb/(2*np.pi*r0)
        #H = 0.0

        dv_x[mask_3b] += (H/8) * (
            -120*np.pi*mu*(cφ**2)*(sθ**3)*sφ
            + 2*np.pi*(12*mu*sθ*sφ + sθv*(np.cos(phv)*(6*np.pi + 4*sθ*sφ) - 3*sφv))
            + cφ * (
                -6*np.pi*mu*(sθ + 5*s3θ)
                + sθ*sθv*(128*sφv - 30*sθ*(-2*np.pi*np.cos(ph0 - phv) + np.sin(ph0 + phv)))
            )
        )


        dv_y[mask_3b] += (H/4) * (
            -3*np.pi*mu*(sθ + 5*s3θ)*sφ
            + 4*cφ*sθ*(np.cos(phv)*sθv + 3*mu*(1 - 5*sθ**2*sφ**2))
            + sθv * (
                -3*np.cos(phv)
                + 30*np.pi*np.cos(ph0 - phv)*sθ**2*sφ
                + 6*np.pi*np.sin(phv)
                + 68*sθ*sφ*np.sin(phv)
                - 15*sθ**2*sφ*np.sin(ph0 + phv)
            )
        )


        dv_z[mask_3b] += (3*H*cθ/4) * (
            -8*np.pi*mu
            + 5 * (
                4*mu*sθ**2*(np.pi - cφ*sφ)
                + 4*sθv*np.sin(phv)
                - sθ*sθv*(-2*np.pi*np.cos(ph0 - phv) + np.sin(ph0 + phv))
            )
        )

        #Hexadecapole
        #dv_x[mask_3b] += norm2*(45/512)*(r_orb/r0)**4*(15 + 28*cos(2*th0) + 21*cos(4*th0))*xhat
        #dv_y[mask_3b] += norm2*(45/512)*(r_orb/r0)**4*(15 + 28*cos(2*th0) + 21*cos(4*th0))*yhat
        #dv_z[mask_3b] += norm2*(15/512)*(r_orb/r0)**4*(29 - 28*cos(2*th0) + 63*cos(4*th0))*zhat
            

    
    vnew_x = vx + dv_x*mult
    vnew_y = vy + dv_y*mult
    vnew_z = vz + dv_z*mult

    vnew_sq = (vnew_x)**2 + (vnew_y)**2 + (vnew_z)**2

    dE += 0.5*(vs**2 - vnew_sq)

    Lnew_x = +(y0*vnew_z - z0*vnew_y)
    Lnew_y = -(x0*vnew_z - z0*vnew_x)
    Lnew_z = +(x0*vnew_y - y0*vnew_x)
    
    dL_z += Lnew_z - Lz

    Lnew = np.sqrt(Lnew_x**2 + Lnew_y**2 + Lnew_z**2)

    dL += Lnew - Ls
    
    return dE, dL, dL_z
    