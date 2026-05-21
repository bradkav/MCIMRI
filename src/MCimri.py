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
DO_INNER   = False
SIMULTANEOUS = False



def calculate_dE(Es, Ls, Lz, binary, r_orb, mult = 1, include_DF=True, include_3body=True, elements="ELLz"):
    
    N = len(Es)
    
    #Calculate some parameters of the binary orbit
    v_orb = binary.v_orb(r_orb)
    Torb = binary.T_orb(r_orb)
    
    m1 = binary.m1
    m2 = binary.m2
    
    #----------------
    b_max = 0.3*r_orb
    r_min = 1e-9*u.pc
    
    r_min_stir = 1.0*r_orb
    
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
    v_rel = v_orb - v_circ

    bs0 = np.sqrt((x0 - x_orb)**2 + (y0 - y_orb)**2 + (z0 - z_orb)**2)
    
    b90  = u.G_N*m2/((v_orb)**2)
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


    mask_DF = bs < 0
    if (include_DF):

        mask_DF  = (b_min < bs) & (bs < b_max)
        
        N_DF = int(np.sum(mask_DF > 0))
        
        if (N_DF > 0):
            
            if (RESAMPLE_B):
                b_new = b_max*np.sqrt(np.random.rand(N_DF))
            else:
                b_new = bs[mask_DF]
                        
            dv_para = 2*v_rel[mask_DF]*(1 + b_new**2/b90**2)**(-1) 
            dv_perp = -2*v_rel[mask_DF]*(b_new/b90)*(1 + b_new**2/b90**2)**(-1) 
            
            dv_x[mask_DF] += dv_para*vorb_x[mask_DF]/v_orb + dv_perp*dx[mask_DF]/bs[mask_DF]
            dv_y[mask_DF] += dv_para*vorb_y[mask_DF]/v_orb + dv_perp*dy[mask_DF]/bs[mask_DF]
            dv_z[mask_DF] += dv_para*vorb_z[mask_DF]/v_orb + dv_perp*dz[mask_DF]/bs[mask_DF]
                    
    if (include_3body):
        
        if (DO_INNER):
            if (SIMULTANEOUS):
                mask_outer = (rs > r_min_stir)
                mask_inner = (rs < r_max_stir)
            else:
                mask_outer = (rs > r_min_stir) & (~mask_DF)
                mask_inner = (rs < r_max_stir) & (~mask_DF)
        
            r0 = rs[mask_outer]
            th0 = theta0[mask_outer]
            ph0 = phi0[mask_outer]
            
            norm = -u.G_N*m2*Torb/r0**2

            xhat = sin(th0)*cos(ph0)
            yhat = sin(th0)*sin(ph0)
            zhat = cos(th0)
            
            dv_x[mask_outer] += norm*(-3/8)*(r/r0)**2*(3 + 5*cos(2*th0))*xhat
            dv_y[mask_outer] += norm*(-3/8)*(r/r0)**2*(3 + 5*cos(2*th0))*yhat
            dv_z[mask_outer] += norm*(+3/4)*(r/r0)**2*(-2 + 5*sin(th0)**2)*zhat
            
            r0 = rs[mask_inner]
            th0 = theta0[mask_inner]
            ph0 = phi0[mask_inner]
            
            norm2 = u.G_N*m2*Torb/a**2

            xhat = sin(th0)*cos(ph0)
            yhat = sin(th0)*sin(ph0)
            zhat = cos(th0)
            
            dv_x[mask_inner] += -norm2*(1/32)*(r0/r)**3*(27 + 45*cos(2*th0))*xhat
            dv_y[mask_inner] += -norm2*(1/32)*(r0/r)**3*(27 + 45*cos(2*th0))*yhat
            dv_z[mask_inner] += norm2*(1/8)*(r0/r)**3*(-3 + 15*cos(2*th0))*zhat
            
        else:
            
            #Add shift due to 3 body effects
            if (SIMULTANEOUS):
                mask_3b = (rs > r_min_stir)
            else:
                mask_3b = (rs > r_min_stir) & (~mask_DF)

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
    