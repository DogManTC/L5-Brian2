"""Brian2 channel equations translated from the original MOD files."""
from __future__ import annotations

import numpy as np
from brian2 import Equations, amp, check_units, cm, exprel, mA, meter, mM, mV, ms, second, siemens, um, volt
from brian2.units.constants import faraday_constant

from .parameters import SimulationParameters

__all__ = ["build_membrane_equations"]


def build_membrane_equations(params: SimulationParameters) -> Equations:
    """Create the complete set of membrane equations for the Brian2 model."""

    qt = 2.3 ** ((34.0 - 21.0) / 10.0)
    eqs = Equations(
        f"""
        Im = I_pas + I_Ih + I_Im + I_Nap + I_NaTa + I_KPst + I_KTst + I_SK + I_SKv3 + I_CaLVA + I_CaHVA - I_inj : amp/meter**2
        I_pas = g_pas*(v - e_pas) : amp/meter**2
        g_pas : siemens/meter**2
        e_pas : volt
        ena : volt
        ek : volt
        eca : volt
        I_inj : amp/meter**2
        qt : 1
        
        # H-current (Ih)
        dm_Ih/dt = (mInf_Ih - m_Ih)/tau_Ih : 1
        mInf_Ih = 1./(1 + exp((v/mV - vh_Ih)/k_Ih)) : 1
        tau_Ih = (1./(exp(-a_Ih - b_Ih*v) + exp(-c_Ih + d_Ih*v)) + e_Ih)*ms : second
        I_Ih = gIhbar_Ih*m_Ih*(v - ehcn_Ih*mV) : amp/meter**2
        gIhbar_Ih : siemens/meter**2
        vh_Ih : 1
        k_Ih : 1
        a_Ih : 1
        b_Ih : 1/volt
        c_Ih : 1
        d_Ih : 1/volt
        e_Ih : 1
        ehcn_Ih : 1
        
        # Im current
        dm_Im/dt = (mInf_Im - m_Im)/tau_Im : 1
        mInf_Im = mAlpha_Im/(mAlpha_Im + mBeta_Im) : 1
        tau_Im = (1/(mAlpha_Im + mBeta_Im))/qt : second
        mAlpha_Im = 3.3e-3/ms*exp(2.5*0.04*(v/mV + 35)) : Hz
        mBeta_Im = 3.3e-3/ms*exp(-2.5*0.04*(v/mV + 35)) : Hz
        I_Im = gImbar_Im*m_Im*(v - ek) : amp/meter**2
        gImbar_Im : siemens/meter**2
        
        # NaTa_t
        dm_NaTa/dt = (mInf_NaTa - m_NaTa)/tau_m_NaTa : 1
        dh_NaTa/dt = (hInf_NaTa - h_NaTa)/tau_h_NaTa : 1
        delta_m_NaTa = (v/mV + 38 + shift_NaTa)/6 : 1
        delta_h_NaTa = (v/mV + 66 + shift_NaTa)/6 : 1
        mAlpha_NaTa = 0.182/ms * 6*exprel(-delta_m_NaTa) : Hz
        mBeta_NaTa = 0.124/ms * 6*exprel(-(-v/mV + (-38 - shift_NaTa))/6) : Hz
        tau_m_NaTa = (1/(mAlpha_NaTa + mBeta_NaTa))/qt : second
        mInf_NaTa = mAlpha_NaTa/(mAlpha_NaTa + mBeta_NaTa) : 1
        hAlpha_NaTa = 0.09/ms * exprel(delta_h_NaTa) : Hz
        hBeta_NaTa = 0.09/ms * exprel((-v/mV + (-66 - shift_NaTa))/6) : Hz
        tau_h_NaTa = (1/(hAlpha_NaTa + hBeta_NaTa))/qt : second
        hInf_NaTa = hAlpha_NaTa/(hAlpha_NaTa + hBeta_NaTa) : 1
        I_NaTa = gNaTa_bar * m_NaTa**3 * h_NaTa * (v - ena) : amp/meter**2
        gNaTa_bar : siemens/meter**2
        shift_NaTa : 1
        
        # Persistent Na (Nap_Et2)
        dm_Nap/dt = (mInf_Nap - m_Nap)/tau_m_Nap : 1
        dh_Nap/dt = (hInf_Nap - h_Nap)/tau_h_Nap : 1
        mInf_Nap = 1.0/(1 + exp((v/mV + 52.6 + shift_Nap)/-4.6)) : 1
        alpha_m_Nap = mAlpha_NaTa : Hz
        beta_m_Nap = mBeta_NaTa : Hz
        tau_m_Nap = 6*(1/(alpha_m_Nap + beta_m_Nap))/qt : second
        hInf_Nap = 1.0/(1 + exp((v/mV + 48.8 + shift_Nap)/10)) : 1
        alpha_h_Nap = (2.88e-6*4.63/ms) * exprel((v/mV + 17 - shift_Nap)/4.63) : Hz
        beta_h_Nap = (6.94e-6*2.63/ms) * exprel(-(v/mV + 64.4 - shift_Nap)/2.63) : Hz
        tau_h_Nap = (1/(alpha_h_Nap + beta_h_Nap))/qt : second
        I_Nap = gNap_bar*m_Nap**3*h_Nap*(v - ena) : amp/meter**2
        gNap_bar : siemens/meter**2
        shift_Nap : 1
        
        # K_Pst
        dm_KPst/dt = (mInf_KPst - m_KPst)/tau_m_KPst : 1
        dh_KPst/dt = (hInf_KPst - h_KPst)/tau_h_KPst : 1
        mInf_KPst = 1/(1 + exp(-(v/mV + 1 + shift_KPst)/12)) : 1
        switch_KPst = 1/(1 + exp((v/mV + 50 + shift_KPst)/0.5)) : 1
        tau_m_KPst = (switch_KPst*(1.25 + 175.03*exp(-(v/mV + shift_KPst)*-0.026)) + (1-switch_KPst)*(1.25 + 13*exp(-(v/mV + shift_KPst)*0.026)))/qt*ms : second
        hInf_KPst = 1/(1 + exp(-(v/mV + 54 + shift_KPst)/-11)) : 1
        tau_h_KPst = (360 + (1010 + 24*(v/mV + 55 + shift_KPst))*exp(-((v/mV + 75 + shift_KPst)/48)**2))/qt*ms : second
        I_KPst = gKPst_bar*m_KPst**2*h_KPst*(v - ek) : amp/meter**2
        gKPst_bar : siemens/meter**2
        shift_KPst : 1
        
        # K_Tst
        dm_KTst/dt = (mInf_KTst - m_KTst)/tau_m_KTst : 1
        dh_KTst/dt = (hInf_KTst - h_KTst)/tau_h_KTst : 1
        mInf_KTst = 1/(1 + exp(-(v/mV)/19)) : 1
        tau_m_KTst = (0.34 + 0.92*exp(-((v/mV + 71)/59)**2))/qt*ms : second
        hInf_KTst = 1/(1 + exp(-(v/mV + 66)/-10)) : 1
        tau_h_KTst = (8 + 49*exp(-((v/mV + 73)/23)**2))/qt*ms : second
        I_KTst = gKTst_bar*m_KTst**4*h_KTst*(v - ek) : amp/meter**2
        gKTst_bar : siemens/meter**2
        
        # SK_E2
        dz_SK/dt = (zInf_SK - z_SK)/tau_z_SK : 1
        zInf_SK = 1/(1 + (0.00043*mM/cai)**4.8) : 1
        tau_z_SK = tau_SK*ms : second
        I_SK = gSK_bar*z_SK*(v - ek) : amp/meter**2
        gSK_bar : siemens/meter**2
        tau_SK : 1
        
        # SKv3_1
        dm_SKv3/dt = (mInf_SKv3 - m_SKv3)/tau_m_SKv3 : 1
        mInf_SKv3 = 1/(1 + exp(((v/mV - (18.7 - shift_SKv3))/(-9.7)))) : 1
        tau_m_SKv3 = 0.2*(20.0/(1 + exp(((v/mV - (-46.56 - shift_SKv3))/(-44.14))))) * ms : second
        I_SKv3 = gSKv3_bar*m_SKv3*(v - ek) : amp/meter**2
        gSKv3_bar : siemens/meter**2
        shift_SKv3 : 1
        
        # Ca_LVAst
        dm_CaLVA/dt = (mInf_CaLVA - m_CaLVA)/tau_m_CaLVA : 1
        dh_CaLVA/dt = (hInf_CaLVA - h_CaLVA)/tau_h_CaLVA : 1
        mInf_CaLVA = 1.0/(1 + exp(((v/mV + 10) - -30)/-6)) : 1
        tau_m_CaLVA = (5.0 + 20.0/(1 + exp(((v/mV + 10) - -25)/5)))/qt*ms : second
        hInf_CaLVA = 1.0/(1 + exp(((v/mV + 10) - -80)/6.4)) : 1
        tau_h_CaLVA = (20.0 + 50.0/(1 + exp(((v/mV + 10) - -40)/7)))/qt*ms : second
        I_CaLVA = gCaLVA_bar*m_CaLVA**2*h_CaLVA*(v - eca) : amp/meter**2
        gCaLVA_bar : siemens/meter**2

        # Ca_HVA
        dm_CaHVA/dt = (mInf_CaHVA - m_CaHVA)/tau_m_CaHVA : 1
        dh_CaHVA/dt = (hInf_CaHVA - h_CaHVA)/tau_h_CaHVA : 1
        mAlpha_CaHVA = 0.055/ms*(-27 - v/mV)/(exp((-27 - v/mV)/3.8) - 1) : Hz
        mBeta_CaHVA = 0.94/ms*exp((-75 - v/mV)/17) : Hz
        mInf_CaHVA = mAlpha_CaHVA/(mAlpha_CaHVA + mBeta_CaHVA) : 1
        tau_m_CaHVA = 1/(mAlpha_CaHVA + mBeta_CaHVA) : second
        hAlpha_CaHVA = 0.000457/ms*exp((-13 - v/mV)/50) : Hz
        hBeta_CaHVA = 0.0065/ms/(exp((-v/mV - 15)/28) + 1) : Hz
        hInf_CaHVA = hAlpha_CaHVA/(hAlpha_CaHVA + hBeta_CaHVA) : 1
        tau_h_CaHVA = 1/(hAlpha_CaHVA + hBeta_CaHVA) : second
        I_CaHVA = gCaHVA_bar*m_CaHVA**2*h_CaHVA*(v - eca) : amp/meter**2
        gCaHVA_bar : siemens/meter**2
        
        # Calcium dynamics
        dcai/dt = -((I_CaLVA + I_CaHVA)*cai_gamma/(2*faraday_constant*depth_cadyn)) - (cai - minCai)/decay_cadyn : mM
        cai_gamma : 1
        depth_cadyn : meter
        minCai : mM
        decay_cadyn : second
        
        """,
        dt=ms,
        substitutions={"qt": qt},
    )
    return eqs
