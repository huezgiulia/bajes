import numpy as np
from ..gw.utils import mcq_to_m1, mcq_to_m2, compute_lambda_tilde

# obs. the following universal constant are not expressed in SI
units_c        = 2.99792458e10     #[cm/s]
units_Msun     = 1.98855e33        #[g]
units_sigma_SB = 5.6704e-5         #[erg/cm^2/s/K^4]
units_h        = 6.6260755e-27     #[erg*s]
units_kB       = 1.380658e-16      #[erg/K]
units_pc2cm    = 3.085678e+18      #[cm/pc]

#
# NR-informed relations
#

def NRfit_recal_mass_dyn(mchirp, q, lambda1, lambda2, NR_fit_recal_mdyn, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_mdyn    = NRfit_log_mass_dyn(mtot, q, lambda1, lambda2) * (1. + NR_fit_recal_mdyn)
    mdyn        = mtot * np.exp(log_mdyn)
    return np.max([0., mdyn])

def NRfit_recal_vel_dyn(mchirp, q, lambda1, lambda2, NR_fit_recal_vdyn, **kwargs):
    mtot    = mchirp / (q/(1+q)**2)**0.6
    vdyn    = NRfit_vel_dyn(mtot, q, lambda1, lambda2) * (1. + NR_fit_recal_vdyn)
    return np.max([1e-10, vdyn])

def NRfit_recal_mass_sec(mchirp, q, lambda1, lambda2, disk_frac_sec, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_m_disk  = NRfit_log_mass_disk(mtot, q, lambda1, lambda2)
    msec        = mtot * np.exp(log_m_disk) * disk_frac_sec
    return np.max([0., msec])

def NRfit_recal_mass_wind(mchirp, q, lambda1, lambda2, disk_frac_wind, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_m_disk  = NRfit_log_mass_disk(mtot, q, lambda1, lambda2)
    mwind       = mtot * np.exp(log_m_disk) * disk_frac_wind
    return np.max([0., mwind])

def NRfit_log_mass_dyn(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for mass of dynamical ejecta
        Returns log(m_ej/M) where M = m1 + m2 (natural log)
    """
    a0, n0, b1, b2, c1, c2 = [-21.295092178221847, 1.9743123050205846,
                              0.0044685525694660435, -0.0024627659648901452,
                              -0.5258273201873834, -0.23928655421412218]

    nu     = q/(1+q)**2
    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_q = 1. + n0*(1.-4.*nu)
    corr_p = 1. + b1*np.sqrt(lambda1) + b2*np.sqrt(lambda2) + c1*m1**(-0.25) + c2*m2**(-0.25)
    return a0 * corr_q * corr_p

def NRfit_vel_dyn(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for velocity of dynamical ejecta
        Returns v_ej / c
    """
    a0, n0, b1, b2, c1, c2 = [0.09217372, -4.52017477, -0.02171866, 0.00946049, -0.2176058, 1.32125944]

    nu     = q/(1+q)**2
    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_q = 1. + n0*(1.-4.*nu)
    corr_p = 1. + b1*np.sqrt(lambda1) + b2*np.sqrt(lambda2) + c1*m1 + c2*m2
    return a0 * corr_q * corr_p

def NRfit_log_mass_disk(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for disk mass
        Returns log(m_disk/M) where M = m1 + m2 (natural log)
    """
    alpha, a1, a2, b1, b2, Lbar, Sbar, Abar = [-13.846080565670077, 4.977942316040381e-06, 
                                               1.8902832916214914e-06, -0.4708068240433623, 
                                               0.33378530243025306, 558.1230475510761, 
                                               -176.21011658144016, 1.0095398304221503]

    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_l = 1. + Abar*((1./np.pi)*np.arctan((lambda1+lambda2-Lbar)/Sbar) - 0.5)
    corr_p = 1 + a1*(lambda1)**2 + a2*(lambda2)**2 + b1*m1**2 + b2*m2**2
    return alpha * corr_l * corr_p

#####
# NR-informed relations "breschi"
#####

def NRfit_recal_mass_dyn_breschi(mchirp, q, lambda1, lambda2, NR_fit_recal_mdyn, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_mdyn    = NRfit_log_mass_dyn_breschi(mtot, q, lambda1, lambda2) * (1. + NR_fit_recal_mdyn)
    mdyn        = mtot * np.exp(log_mdyn)
    return np.max([0., mdyn])

def NRfit_recal_vel_dyn_breschi(mchirp, q, lambda1, lambda2, NR_fit_recal_vdyn, **kwargs):
    mtot    = mchirp / (q/(1+q)**2)**0.6
    vdyn    = NRfit_vel_dyn_breschi(mtot, q, lambda1, lambda2) * (1. + NR_fit_recal_vdyn)
    return np.max([1e-10, vdyn])

def NRfit_recal_mass_sec_breschi(mchirp, q, lambda1, lambda2, disk_frac_sec, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_m_disk  = NRfit_log_mass_disk_breschi(mtot, q, lambda1, lambda2)
    msec        = mtot * np.exp(log_m_disk) * disk_frac_sec
    return np.max([0., msec])

def NRfit_recal_mass_wind_breschi(mchirp, q, lambda1, lambda2, disk_frac_wind, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_m_disk  = NRfit_log_mass_disk_breschi(mtot, q, lambda1, lambda2)
    mwind       = mtot * np.exp(log_m_disk) * disk_frac_wind
    return np.max([0., mwind])

def NRfit_log_mass_dyn_breschi(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for mass of dynamical ejecta
        Returns log(m_ej/M) where M = m1 + m2 (natural log)
    """ 
    a0, n0, b1, b2, c1, c2 = [-128.41717311068592, 16.111485804150124,
                              0.001072666803983363, -0.0005299835909359168,
                              -0.4939247225352364, -0.5383287345344886]

    nu     = q/(1+q)**2
    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_q = 1. + n0*(1.-4.*nu)
    corr_p = 1. + b1*np.sqrt(lambda1) + b2*np.sqrt(lambda2) + c1*m1**(-0.25) + c2*m2**(-0.25)
    return a0 * corr_q * corr_p

def NRfit_vel_dyn_breschi(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for velocity of dynamical ejecta
        Returns v_ej / c
    """
    a0, n0, b1, b2, c1, c2 = [0.09885394177197987, -6.581266594480056, -0.03906332013176158, 
                              -0.03004155486380879, 1.1913263819216484, 0.6699361738637798] 
    
    nu     = q/(1+q)**2
    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_q = 1. + n0*(1.-4.*nu)
    corr_p = 1. + b1*np.sqrt(lambda1) + b2*np.sqrt(lambda2) + c1*m1 + c2*m2
    return a0 * corr_q * corr_p

def NRfit_log_mass_disk_breschi(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for disk mass
        Returns log(m_disk/M) where M = m1 + m2 (natural log)
    """
    alpha, a1, a2, b1, b2, Lbar, Sbar, Abar = [-1.00005048163444, -6.30219031649512e-07, 
                                               -1.5709263623799608e-07, -0.25794191019918405, 
                                               -0.5842440314342542, 3.1839377906331714e-05, 
                                               954.9345228010245, 23.12026340504407] 

    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_l = 1. + Abar*((1./np.pi)*np.arctan((lambda1+lambda2-Lbar)/Sbar) - 0.5) 
    corr_p = 1 + a1*(lambda1)**2 + a2*(lambda2)**2 + b1*m1**2 + b2*m2**2
    return alpha * corr_l * corr_p

#####
# NR-informed relations "breschi" as sum of the components
#####

def NRfit_recal_mass_dyn_sum_breschi(mchirp, q, lambda1, lambda2, disk_frac_sec, frac, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_mdyn    = NRfit_log_mass_dyn_breschi(mtot, q, lambda1, lambda2)
    mdyn_fit    = mtot * np.exp(log_mdyn)
    log_m_disk  = NRfit_log_mass_disk_breschi(mtot, q, lambda1, lambda2)
    msec_fit    = mtot * np.exp(log_m_disk) * disk_frac_sec
    mdyn        = frac * (mdyn_fit + msec_fit)
    return np.max([0., mdyn])

def NRfit_recal_mass_sec_sum_breschi(mchirp, q, lambda1, lambda2, disk_frac_sec, frac, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_mdyn    = NRfit_log_mass_dyn_breschi(mtot, q, lambda1, lambda2)
    mdyn_fit    = mtot * np.exp(log_mdyn)
    log_m_disk  = NRfit_log_mass_disk_breschi(mtot, q, lambda1, lambda2)
    msec_fit    = mtot * np.exp(log_m_disk) * disk_frac_sec
    msec        = (1 - frac) * (mdyn_fit + msec_fit)
    return np.max([0., msec])

#####
# NR-informed relations "nedora"
#####

from bajes.obs.gw.utils import compute_lambda_tilde

def NRfit_recal_mass_dyn_nedora(mchirp, q, lambda1, lambda2, **kwargs): #NR_fit_recal_mdyn
    mtot        = mchirp / (q/(1+q)**2)**0.6
    m1          = mtot*q/(1+q)
    m2          = mtot/(1+q)
    lt          = compute_lambda_tilde(m1, m2, lambda1, lambda2)
    mdyn10_3    = NRfit_log_mass_dyn_nedora(q, lt) #* (1. + NR_fit_recal_mdyn)
    mdyn        = mdyn10_3*1e-3
    return np.max([0., mdyn])

def NRfit_recal_vel_dyn_nedora(mchirp, q, lambda1, lambda2, **kwargs):  #NR_fit_recal_vdyn
    mtot        = mchirp / (q/(1+q)**2)**0.6
    m1          = mtot*q/(1+q)
    m2          = mtot/(1+q)
    lt          = compute_lambda_tilde(m1, m2, lambda1, lambda2)
    vdyn    = NRfit_vel_dyn_nedora(q, lt) #* (1. + NR_fit_recal_vdyn)
    return np.max([1e-10, vdyn])

def NRfit_recal_mass_sec_nedora(mchirp, q, lambda1, lambda2, disk_frac_sec, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    m1          = mtot*q/(1+q)
    m2          = mtot/(1+q)
    lt          = compute_lambda_tilde(m1, m2, lambda1, lambda2)
    m_disk      = NRfit_log_mass_disk_nedora(q, lt)
    msec        = m_disk * disk_frac_sec
    return np.max([0., msec])

def NRfit_recal_mass_wind_nedora(mchirp, q, lambda1, lambda2, disk_frac_wind, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    m1          = mtot*q/(1+q)
    m2          = mtot/(1+q)
    lt          = compute_lambda_tilde(m1, m2, lambda1, lambda2)
    m_disk      = NRfit_log_mass_disk_nedora(q, lt)
    mwind       = m_disk * disk_frac_wind
    return np.max([0., mwind])

def NRfit_log_mass_dyn_nedora(q, lt):
    """
        NR-calibrated relation for mass of dynamical ejecta
        Returns log(m_ej/M) where M = m1 + m2 (natural log) - 
        Residuals = 20.360343174184003 e St Dev = 0.7912917303721327
    """
    b0, b1, b2, b3, b4, b5 = [19.11486051225742, -5.810359396825988,
                              -0.054258334940957276, -2.528180825757026,
                              0.028827084277108, 1.6408045303840963e-05]

    val = b0 + b1*q + b2*lt + b3*q**2 + b4*q*lt + b5*lt**2
    return val

def NRfit_vel_dyn_nedora(q, lt):
    """
        NR-calibrated relation for velocity of dynamical ejecta
        Returns v_ej / c -
        Residuals = 0.25913242031990996 e St Dev = 0.09118396501754546
    """
    b0, b1, b2, b3, b4, b5 = [0.3245584679500216, 0.3037063354193521,
                              -0.0009362814001789095, -0.17383847811798092,
                              0.00011142463805362791, 5.840741169097888e-07]

    val = b0 + b1*q + b2*lt + b3*q**2 + b4*q*lt + b5*lt**2
    return val

def NRfit_log_mass_disk_nedora(q, lt):
    """
        NR-calibrated relation for disk mass
        Returns log(m_disk/M) where M = m1 + m2 (natural log)
    """
    b0, b1, b2, b3, b4, b5 = [-1.0065191726047198, 1.142069190213931,
                              0.0008123152157553065, -0.35127395941697265,
                              0.0001711936647410558, -6.935939552742011e-07]

    val = b0 + b1*q + b2*lt + b3*q**2 + b4*q*lt + b5*lt**2
    return val


def compute_integral(E0, theta_C, theta_w, N=50):
    # Gauss-Legendre nodes and weights
    x, w = np.polynomial.legendre.leggauss(N)

    # Map from x in [-1, 1] to theta in [0, theta_w]
    theta = 0.5 * theta_w * (x + 1)

    E_theta = E0 * np.exp(-0.5 * (theta / theta_C)**2)
    integrand = E_theta * np.sin(theta)
    integral = 2 * np.pi * 0.5 * theta_w * np.dot(w, integrand)

    return integral

def joint_rel_mdisc(thetaCore, E0, thetaWing, disc_sec_frac, **kwargs):
    """
        Relation to connect the disc mass
        to the isotropic energy of the GRB
        for Gaussian jet
        https://arxiv.org/abs/2006.07376
    """
    eta = 0.6e-3
    fw = 0.3
    energy = compute_integral(E0, thetaCore, thetaWing)
    mdisc  = energy * thetaCore**2 / (2 * eta * (1 - fw))
    return mdisc * disc_sec_frac

if __name__ == "__main__":
    m_ej = NRfit_recal_mass_dyn(1.1852778957839742, 2.472884198170911, 839.7030363651173, 3465.539569009505)
    print(m_ej)

# New NR fits 12/25

def NRfit_recal_mass_dyn_new(mchirp, q, lambda1, lambda2, NR_fit_recal_mdyn, **kwargs):
    lambdat = compute_lambda_tilde(mcq_to_m1(mchirp, q), mcq_to_m2(mchirp, q), lambda1, lambda2)
    mdyn = NRfit_mass_dyn_new(lambdat, q) * (1. + NR_fit_recal_mdyn)
    return np.max([0., mdyn])

def NRfit_mass_dyn_new(lt, q):
    """
        NR-calibrated relation for mass of dynamical ejecta

        Residuals = 4.177388774744708
        St Dev    = 0.30462433835604796
        Chi2      =  104.4347193686177
    """
    l0 = (lt - 338) / 338
    q0 = (1 - 1 / q)
    a, b0, b1, c0, c1 = [0.005465415864767121, 2.7432092625399895, -0.5095276905516682, 
                         -0.004654444440538087, 3.4572418092166997]
    return a * (1 + (c0 / a + c1 * l0) * q0) / (1 + b0 * l0 + b1 * l0**2)

def NRfit_recal_vel_dyn_new(mchirp, q, lambda1, lambda2, NR_fit_recal_vdyn, **kwargs):
    lambdat = compute_lambda_tilde(mcq_to_m1(mchirp, q), mcq_to_m2(mchirp, q), lambda1, lambda2)
    vdyn    = NRfit_vel_dyn_new(lambdat, q) * (1. + NR_fit_recal_vdyn)
    return np.max([1e-10, vdyn])

def NRfit_vel_dyn_new(lt, q):
    """
        NR-calibrated relation for velocity of dynamical ejecta
        Returns v_ej / c

        Residuals = 0.537204002095301
        St Dev    = 0.11250984191094857
        Chi2 =  13.430100052382517
    """
    l0 = (lt - 338) / 338
    q0 = (1 - 1 / q)
    a, b0, b1, c0, c1 = [0.25426451470687095, 0.8571917872951094, -0.27004689267758164,
                         -0.21143849689323335, 0.7419773536876892]
    return a * (1 + (c0 / a + c1 * l0) * q0) / (1 + b0 * l0 + b1 * l0**2)

def NRfit_recal_mass_sec_new(mchirp, q, lambda1, lambda2, disk_frac_sec, **kwargs):
    lambdat = compute_lambda_tilde(mcq_to_m1(mchirp, q), mcq_to_m2(mchirp, q), lambda1, lambda2)
    log_m_disk  = NRfit_log_mass_disk_new(lambdat, q)
    msec        = np.exp(log_m_disk) * disk_frac_sec
    return np.max([0., msec])

def NRfit_log_mass_disk_new(lt, q):
    """
        NR-calibrated relation for mass of disk mass
        Returns log_10(m_disk)

        Residuals = 1.5980486826798666
        St Dev    = 0.19696608880216374
        Chi2 =  39.951217066996634
    """
    l0 = (lt - 338) / 338
    q0 = (1 - 1 / q)
    a, b, c, d = [-0.6571971321505049, -2.810766313956197,
                  1.3857706576461155, 15.363897209682197]
    return 1 / (a + b * q0) + np.tanh(l0) / (c + d * q0)