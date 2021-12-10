#Dynamic Optimization to maximise a representative agent Welfare Function over a given period. Extends Nordhaus' model to account for climate change-driven biodiversity loss and relative price changes. 


"""
Created on Fri Feb 26 17:19:58 2021

@author: Tommaso
"""
# -*- coding: utf-8 -*-

import numpy as np
import time
from numba import njit,guvectorize,float64
import scipy.optimize as opt
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os

data = {}

os.chdir('C:/Users/Tommaso/Desktop/Masterarbeit/PyDICE')
tot_time=0 #start time counter
Time_horizon=100 #Insert Plots Time Horizon
beta=0.1
zeta=-0.11
c0=77.605649  #10.483    
phi=0.5
pi_list=[-2.5,1]
bex=1.58
k= 1.73*10**-2
fi=4.4*10**-3
teta= 2.8*10**-4
ES0=c0
O0=c0
EQbar=0.1*(ES0)

for pi in pi_list:

    B3=(teta+k*3+fi*3**2)
    x=(1-beta)*((1-0.0326)*c0)**zeta-(1-beta)*((1-0.0163)*c0)**zeta+beta*((phi*(ES0)**pi+(1-phi)*O0**pi)**(1/pi)-EQbar)**zeta
    psi=(1/9)*(((((1/(1-phi))*(((x/beta)**(1/zeta)+EQbar)**pi-phi*((1-0.0036)*ES0)**pi))**(1/pi))**(-1)*O0)-1)
    ro=-(1/B3**bex)*((1/ES0)*((1/phi)*(((x/beta)**(1/zeta)+EQbar)**pi-(1-phi)*(O0/(1+psi*9))**pi))**(1/pi)-1)

    print(psi)
    print(ro)
    
    t = np.arange(1, 101)
    NT = len(t)
    
    #Parameters
    fosslim = 6000 # Maximum cumulative extraction fossil fuels (GtC); denoted by Cum
    tstep  = 5 # Years per Period
    ifopt  = 0 # Indicator where optimized is 1 and base is 0
    
    #Preferences
    
    elasmu = 1.35 # 1.45  Elasticity of marginal utility of consumption
    prstp = 0.011 # 0.015  Initial rate of social time preference per year 
    
    #** Population and technology
    gama  = 0.300 #   Capital elasticity in production function         /.300 /
    pop0  = 7403   # Initial world population 2015 (millions)          /7403 /
    popadj = 0.134 #  Growth rate to calibrate to 2050 pop projection  /0.134/
    popasym = 11500 # Asymptotic population (millions)                 /11500/
    dk  = 0.100 #     Depreciation rate on capital (per year)           /.100 /
    q0  = 105.5 #     Initial world gross output 2015 (trill 2010 USD) /105.5/
    k0  = 223 #     Initial capital value 2015 (trill 2010 USD)        /223  /
    a0  = 5.115 #     Initial level of total factor productivity       /5.115/
    ga0  = 0.076 #    Initial growth rate for TFP per 5 years          /0.076/
    dela  = 0.005 #   Decline rate of TFP per 5 years                  /0.005/
    
    #** Emissions parameters
    gsigma1  = -0.0152 # Initial growth of sigma (per year)            /-0.0152/
    dsig  = -0.001 #   Decline rate of decarbonization (per period)    /-0.001 /
    eland0 = 2.6 #  Carbon emissions from land 2015 (GtCO2 per year)   / 2.6   /
    deland = 0.115 # Decline rate of land emissions (per period)        / .115  /
    e0 = 35.85 #    Industrial emissions 2015 (GtCO2 per year)       /35.85  /
    miu0  = 0.03 #   Initial emissions control rate for base case 2015  /.03    /
    
    #** Carbon cycle
    #* Initial Conditions
    mat0 = 851 #  Initial Concentration in atmosphere 2015 (GtC)       /851  /
    mu0  = 460 #  Initial Concentration in upper strata 2015 (GtC)     /460  /
    ml0  = 1740 #  Initial Concentration in lower strata 2015 (GtC)    /1740 /
    mateq = 588 # mateq Equilibrium concentration atmosphere  (GtC)    /588  /
    mueq  = 360 # mueq Equilibrium concentration in upper strata (GtC) /360  /
    mleq = 1720 # mleq Equilibrium concentration in lower strata (GtC) /1720 /
    
    #* Flow paramaters, denoted by Phi_ij in the model
    b12  = 0.12 #    Carbon cycle transition matrix                     /.12  /
    b23  = 0.007 #   Carbon cycle transition matrix                    /0.007/
    #* These are for declaration and are defined later
    b11  = None   # Carbon cycle transition matrix
    b21  = None  # Carbon cycle transition matrix
    b22  = None  # Carbon cycle transition matrix
    b32  = None  # Carbon cycle transition matrix
    b33  = None  # Carbon cycle transition matrix
    sig0  = None  # Carbon intensity 2010 (kgCO2 per output 2005 USD 2010)
    
    #** Climate model parameters
    t2xco2  = 3.1 # Equilibrium temp impact (oC per doubling CO2)    / 3.1 /
    fex0  = 0.5 #   2015 forcings of non-CO2 GHG (Wm-2)              / 0.5 /
    fex1  = 1.0 #   2100 forcings of non-CO2 GHG (Wm-2)              / 1.0 /
    tocean0  = 0.0068 # Initial lower stratum temp change (C from 1900) /.0068/
    tatm0  = 0.85 #  Initial atmospheric temp change (C from 1900)    /0.85/
    c1  = 0.1005 #     Climate equation coefficient for upper level  /0.1005/
    c3  = 0.088 #     Transfer coefficient upper to lower stratum    /0.088/
    c4  = 0.025 #     Transfer coefficient for lower level           /0.025/
    fco22x  = 3.6813 # eta in the model; Eq.22 : Forcings of equilibrium CO2 doubling (Wm-2)   /3.6813 /
    
    #** Climate damage parameters
    a10  = 0 #     Initial damage intercept                         /0   /
    a20  = None #     Initial damage quadratic term
    a1  = 0 #      Damage intercept                                 /0   /
    a2  = 0.00181 #      Damage quadratic term                     /0.00236/
    a3  = 2.00 #      Damage exponent                              /2.00   /
    
    #** Abatement cost
    expcost2 = 2.6 # Theta2 in the model, Eq. 10 Exponent of control cost function             / 2.6  /
    pback  = 550 #   Cost of backstop 2010$ per tCO2 2015          / 550  /
    gback  = 0.025 #   Initial cost decline backstop cost per period / .025/
    limmiu  = 1.2 #  Upper limit on control rate after 2150        / 1.2 /
    tnopol  = 45 #  Period before which no emissions controls base  / 45   /
    cprice0  = 2 # Initial base carbon price (2010$ per tCO2)      / 2    /
    gcprice  = 0.02 # Growth rate of base carbon price per year     /.02  /
    
    #** Scaling and inessential parameters
    #* Note that these are unnecessary for the calculations
    #* They ensure that MU of first period's consumption =1 and PV cons = PV utilty
    scale1  = 0.0302455265681763 #    Multiplicative scaling coefficient           /0.0302455265681763 /
    scale2  = -10993.704 #    Additive scaling coefficient       /-10993.704/;
    
    #* Parameters for long-run consistency of carbon cycle 
    #(Question)
    b11 = 1 - b12
    b21 = b12*mateq/mueq
    b22 = 1 - b21 - b23
    b32 = b23*mueq/mleq
    b33 = 1 - b32
    
    #* Further definitions of parameters
    a20 = a2
    sig0 = e0/(q0*(1-miu0)) #From Eq. 14
    lam = fco22x/ t2xco2 #From Eq. 25
    
    l = np.zeros(NT)
    l[0] = pop0 #Labor force
    al = np.zeros(NT) 
    al[0] = a0
    gsig = np.zeros(NT) 
    gsig[0] = gsigma1
    sigma = np.zeros(NT)
    sigma[0]= sig0
    ga = ga0 * np.exp(-dela*5*(t-1)) #TFP growth rate dynamics, Eq. 7
    pbacktime = pback * (1-gback)**(t-1) #Backstop price
    etree = eland0*(1-deland)**(t-1) #Emissions from deforestration
    rr = 1/((1+prstp)**(tstep*(t-1))) #Eq. 3
    #The following three equations define the exogenous radiative forcing; used in Eq. 23  
    forcoth = np.full(NT,fex0)
    forcoth[0:18] = forcoth[0:18] + (1/17)*(fex1-fex0)*(t[0:18]-1)
    forcoth[18:NT] = forcoth[18:NT] + (fex1-fex0)
    optlrsav = (dk + .004)/(dk + .004*elasmu + prstp)*gama #Optimal long-run savings rate used for transversality (Question)
    cost1 = np.zeros(NT)
    cumetree = np.zeros(NT)
    cumetree[0] = 100
    cpricebase = cprice0*(1+gcprice)**(5*(t-1)) 
    
    @njit('(float64[:], int32)')
    def InitializeLabor(il,iNT):
        for i in range(1,iNT):
            il[i] = il[i-1]*(popasym / il[i-1])**popadj
    
    @njit('(float64[:], int32)')        
    def InitializeTFP(ial,iNT):
        for i in range(1,iNT):
            ial[i] = ial[i-1]/(1-ga[i-1])
            
    @njit('(float64[:], int32)')        
    def InitializeGrowthSigma(igsig,iNT):
        for i in range(1,iNT):
            igsig[i] = igsig[i-1]*((1+dsig)**tstep)
            
    @njit('(float64[:], float64[:],float64[:],int32)')        
    def InitializeSigma(isigma,igsig,icost1,iNT):
        for i in range(1,iNT):
            isigma[i] =  isigma[i-1] * np.exp(igsig[i-1] * tstep)
            icost1[i] = pbacktime[i] * isigma[i]  / expcost2 /1000
            
    @njit('(float64[:], int32)')        
    def InitializeCarbonTree(icumetree,iNT):
        for i in range(1,iNT):
            icumetree[i] = icumetree[i-1] + etree[i-1]*(5/3.666)

    """
    First: Functions related to emissions of carbon and weather damages
    """
    
    # Retuns the total carbon emissions; Eq. 18
    @njit('float64(float64[:],int32)') 
    def fE(iEIND,index):
        return iEIND[index] + etree[index]
    
    #Eq.14: Determines the emission of carbon by industry EIND
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def fEIND(iYGROSS, iMIU, isigma,index):
        return isigma[index] * iYGROSS[index] * (1 - iMIU[index])
    
    #Cumulative industrial emission of carbon
    @njit('float64(float64[:],float64[:],int32)') 
    def fCCA(iCCA,iEIND,index):
        return iCCA[index-1] + iEIND[index-1] * 5 / 3.666
    
    #Cumulative total carbon emission
    @njit('float64(float64[:],float64[:],int32)')
    def fCCATOT(iCCA,icumetree,index):
        return iCCA[index] + icumetree[index]
    
    #Eq. 22: the dynamics of the radiative forcing
    @njit('float64(float64[:],int32)')
    def fFORC(iMAT,index):
        return fco22x * np.log(iMAT[index]/588.000)/np.log(2) + forcoth[index]
    
    # Dynamics of Omega; Eq.9
    @njit('float64(float64[:],int32)')
    def fDAMFRAC(iTATM,index):
        return a1*iTATM[index] + a2*iTATM[index]**a3
    
    #Calculate damages as a function of Gross industrial production; Eq.8 
    @njit('float64(float64[:],float64[:],int32)')
    def fDAMAGES(iYGROSS,iDAMFRAC,index):
        return iYGROSS[index] * iDAMFRAC[index]
    
    #Dynamics of Lambda; Eq. 10 - cost of the reudction of carbon emission (Abatement cost)
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def fABATECOST(iYGROSS,iMIU,icost1,index):
        return iYGROSS[index] * icost1[index] * iMIU[index]**expcost2
    
    #Marginal Abatement cost
    @njit('float64(float64[:],int32)')
    def fMCABATE(iMIU,index):
        return pbacktime[index] * iMIU[index]**(expcost2-1)
    
    #Price of carbon reduction
    @njit('float64(float64[:],int32)')
    def fCPRICE(iMIU,index):
        return pbacktime[index] * (iMIU[index])**(expcost2-1)
    
    #Eq. 19: Dynamics of the carbon concentration in the atmosphere
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def fMAT(iMAT,iMU,iE,index):
        if(index == 0):
            return mat0
        else:
            return iMAT[index-1]*b11 + iMU[index-1]*b21 + iE[index-1] * 5 / 3.666
    
    #Eq. 21: Dynamics of the carbon concentration in the ocean LOW level
    @njit('float64(float64[:],float64[:],int32)') 
    def fML(iML,iMU,index):
        if(index == 0):
            return ml0
        else:
            return iML[index-1] * b33  + iMU[index-1] * b23
    
    #Eq. 20: Dynamics of the carbon concentration in the ocean UP level
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def fMU(iMAT,iMU,iML,index):
        if(index == 0):
            return mu0
        else:
            return iMAT[index-1]*b12 + iMU[index-1]*b22 + iML[index-1]*b32
    
    #Eq. 23: Dynamics of the atmospheric temperature
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def fTATM(iTATM,iFORC,iTOCEAN,index):
        if(index == 0):
            return tatm0
        else:
            return iTATM[index-1] + c1 * (iFORC[index] - (fco22x/t2xco2) * iTATM[index-1] - c3 * (iTATM[index-1] - iTOCEAN[index-1]))
    
    #Eq. 24: Dynamics of the ocean temperature
    @njit('float64(float64[:],float64[:],int32)')
    def fTOCEAN(iTATM,iTOCEAN,index):
        if(index == 0):
            return tocean0
        else:
            return iTOCEAN[index-1] + c4 * (iTATM[index-1] - iTOCEAN[index-1])
    
    """
    Second: Function related to economic variables
    """
    
    #The total production without climate losses denoted previously by YGROSS
    @njit('float64(float64[:],float64[:],float64[:],int32)')
    def fYGROSS(ial,il,iK,index):
        return ial[index] * ((il[index]/1000)**(1-gama)) * iK[index]**gama
    
    #The production under the climate damages cost
    @njit('float64(float64[:],float64[:],int32)')
    def fYNET(iYGROSS, iDAMFRAC, index):
        return iYGROSS[index] * (1 - iDAMFRAC[index])
    
    #Production after abatement cost
    @njit('float64(float64[:],float64[:],int32)')
    def fY(iYNET,iABATECOST,index):
        return iYNET[index] - iABATECOST[index]
    
    #Consumption Eq. 11
    @njit('float64(float64[:],float64[:],int32)')
    def fC(iY,iI,index):
        if (index == 0):
            return c0
        else:
            return iY[index] - iI[index]
    
    #Per capita consumption, Eq. 12
    @njit('float64(float64[:],float64[:],int32)')
    def fCPC(iC,il,index):
        return 1000 * iC[index] / il[index]
    
    #Saving policy: investment
    @njit('float64(float64[:],float64[:],int32)')
    def fI(iS,iY,index):
        return iS[index] * iY[index] 
    
    #Capital dynamics Eq. 13
    @njit('float64(float64[:],float64[:],int32)')
    def fK(iK,iI,index):
        if(index == 0):
            return k0
        else:
            return (1-dk)**tstep * iK[index-1] + tstep * iI[index-1]
    
    #Interest rate equation; Eq. 26 added in personal notes
    @njit('float64(float64[:],int32)')
    def fRI(iCPC,index):
        return (1 + prstp) * (iCPC[index+1]/iCPC[index])**(elasmu/tstep) - 1
    
    #Periodic utility: A form of Eq. 2
    @njit('float64(float64[:],float64[:],int32)')
    def fCEMUTOTPER(iPERIODU,il,index):
        return iPERIODU[index] * il[index] * rr[index]

            
    @njit('float64(float64[:],int32)')
    def fB(iTATM, index):
        return (teta+k*iTATM[index]+fi*iTATM[index]**2)
                
    @njit('float64(float64[:],int32)')
    def fES(iB, index):
        return ES0*(1-ro*iB[index]**bex)
       
    
    @njit('float64(float64[:],float64[:],int32)')
    def fO(iO, iTATM, index):
        if (index==0):
            return O0
        else:
            return iO[0]/(1+psi*iTATM[index]**2)        

    @njit('float64(float64[:],float64[:],int32)')
    def fEQ(iES,iO,index):
        return (phi*iES[index]**pi+(1-phi)*iO[index]**pi)**(1/pi)
    
    
    #The term between brackets in Eq. 2
    @njit('float64(float64[:],float64[:],float64[:],int32)')
    def fPERIODU(iC,iEQ,il,index):
        return (((1-beta)*(1000/il[index]*iC[index])**zeta+beta*((iEQ[index]-EQbar)*1000/il[index])**(zeta))**((1-elasmu)/zeta))/(1-elasmu)

    
    #utility function
    @guvectorize([(float64[:], float64[:])], '(n), (m)')
    def fUTILITY(iCEMUTOTPER, resUtility):
        resUtility[0] = tstep *np.sum(iCEMUTOTPER) #tstep * scale1 * np.sum(iCEMUTOTPER) + scale2 #scale1 * np.sum(iCEMUTOTPER) #
    
    """
    In this part we implement the objective function
    """
    
    # * Control rate limits
    MIU_lo = np.full(NT,0.01)
    MIU_up = np.full(NT,limmiu)
    MIU_up[0:29] = 1
    MIU_lo[0] = miu0
    MIU_up[0] = miu0
    MIU_lo[MIU_lo==MIU_up] = 0.99999*MIU_lo[MIU_lo==MIU_up]
    bnds1=[]
    for i in range(NT):
        bnds1.append((MIU_lo[i],MIU_up[i]))
    # * Control variables
    lag10 = t > NT - 10
    S_lo = np.full(NT,1e-1)
    S_lo[lag10] = optlrsav
    S_up = np.full(NT,0.9)
    S_up[lag10] = optlrsav
    S_lo[S_lo==S_up] = 0.99999*S_lo[S_lo==S_up]
    bnds2=[]
    for i in range(NT):
        bnds2.append((S_lo[i],S_up[i]))
        
    # Arbitrary starting values for the control variables:
    S_start = np.full(NT,0.2)
    S_start[S_start < S_lo] = S_lo[S_start < S_lo]
    S_start[S_start > S_up] = S_lo[S_start > S_up]
    MIU_start = 0.2*MIU_up
    MIU_start[MIU_start < MIU_lo] = MIU_lo[MIU_start < MIU_lo]
    MIU_start[MIU_start > MIU_up] = MIU_up[MIU_start > MIU_up]
    
    K = np.zeros(NT)
    YGROSS = np.zeros(NT)
    EIND = np.zeros(NT)
    E = np.zeros(NT)
    CCA = np.zeros(NT)
    CCATOT = np.zeros(NT)
    MAT = np.zeros(NT)
    ML = np.zeros(NT)
    MU = np.zeros(NT)
    FORC = np.zeros(NT)
    TATM = np.zeros(NT)
    TOCEAN = np.zeros(NT)
    DAMFRAC = np.zeros(NT)
    DAMAGES = np.zeros(NT)
    ABATECOST = np.zeros(NT)
    MCABATE = np.zeros(NT)
    CPRICE = np.zeros(NT)
    YNET = np.zeros(NT)
    Y = np.zeros(NT)
    I = np.zeros(NT)
    C = np.zeros(NT)
    EQ=np.zeros(NT)
    Elim=np.zeros(NT)
    CPC = np.zeros(NT)
    RI = np.zeros(NT)
    PERIODU = np.zeros(NT)
    CEMUTOTPER = np.zeros(NT)
    B=np.zeros(NT)
    ES=np.zeros(NT)
    O=np.zeros(NT)
    #The objective function
    #It returns the utility as scalar
    def fOBJ(x,sign,iI,iK,ial,il,iYGROSS,isigma,iEIND,iE,iCCA,iCCATOT,icumetree,iMAT,iMU,iML,iFORC,iTATM,iTOCEAN,iDAMFRAC,iDAMAGES,iABATECOST,icost1,iMCABATE,
             iCPRICE,iYNET,iY,iC,iB,iES,iO,iEQ,iCPC,iPERIODU,iCEMUTOTPER,iRI,iNT):
        
        iMIU = x[0:NT]
        iS = x[NT:(2*NT)]
        
        for i in range(iNT):
            iK[i] = fK(iK,iI,i)
            iYGROSS[i] = fYGROSS(ial,il,iK,i)
            iEIND[i] = fEIND(iYGROSS, iMIU, isigma,i)
            iE[i] = fE(iEIND,i)
            iCCA[i] = fCCA(iCCA,iEIND,i)
            iCCATOT[i] = fCCATOT(iCCA,icumetree,i)
            iMAT[i] = fMAT(iMAT,iMU,iE,i)
            iML[i] = fML(iML,iMU,i)
            iMU[i] = fMU(iMAT,iMU,iML,i)
            iFORC[i] = fFORC(iMAT,i)
            iTATM[i] = fTATM(iTATM,iFORC,iTOCEAN,i)
            iTOCEAN[i] = fTOCEAN(iTATM,iTOCEAN,i)
            iDAMFRAC[i] = fDAMFRAC(iTATM,i)
            iDAMAGES[i] = fDAMAGES(iYGROSS,iDAMFRAC,i)
            iABATECOST[i] = fABATECOST(iYGROSS,iMIU,icost1,i)
            iMCABATE[i] = fMCABATE(iMIU,i)
            iCPRICE[i] = fCPRICE(iMIU,i)
            iYNET[i] = fYNET(iYGROSS, iDAMFRAC, i)
            iY[i] = fY(iYNET,iABATECOST,i)
            iI[i] = fI(iS,iY,i)
            iC[i] = fC(iY,iI,i)
            iB[i]=fB(iTATM,i)
            iES[i]=fES(iB,i)
            iO[i]=fO(iO,iTATM,i)
            iEQ[i]=fEQ(iES,iO,i)
            iCPC[i] = fCPC(iC,il,i)
            iPERIODU[i] = fPERIODU(iC,iEQ,il,i)
            iCEMUTOTPER[i] = fCEMUTOTPER(iPERIODU,il,i)
            iRI = fRI(iCPC,i)
            
        resUtility = np.zeros(1)
        fUTILITY(iCEMUTOTPER, resUtility)
        
        return sign*resUtility[0]
    
    #For the optimal allocation of x, calculates the whole system variables
    def Optimality(x,iI,iK,ial,il,iYGROSS,isigma,iEIND,iE,iCCA,iCCATOT,icumetree,iMAT,iMU,iML,iFORC,iTATM,iTOCEAN,iDAMFRAC,iDAMAGES,iABATECOST,icost1,iMCABATE,
             iCPRICE,iYNET,iY,iC,iB,iES,iO,iEQ,iCPC,iPERIODU,iCEMUTOTPER,iRI,iNT):
        
        iMIU = x[0:NT]
        iS = x[NT:(2*NT)]
        
        for i in range(iNT):
            iK[i] = fK(iK,iI,i)
            iYGROSS[i] = fYGROSS(ial,il,iK,i)
            iEIND[i] = fEIND(iYGROSS, iMIU, isigma,i)
            iE[i] = fE(iEIND,i)
            iCCA[i] = fCCA(iCCA,iEIND,i)
            iCCATOT[i] = fCCATOT(iCCA,icumetree,i)
            iMAT[i] = fMAT(iMAT,iMU,iE,i)
            iML[i] = fML(iML,iMU,i)
            iMU[i] = fMU(iMAT,iMU,iML,i)
            iFORC[i] = fFORC(iMAT,i)
            iTATM[i] = fTATM(iTATM,iFORC,iTOCEAN,i)
            iTOCEAN[i] = fTOCEAN(iTATM,iTOCEAN,i)
            iDAMFRAC[i] = fDAMFRAC(iTATM,i)
            iDAMAGES[i] = fDAMAGES(iYGROSS,iDAMFRAC,i)
            iABATECOST[i] = fABATECOST(iYGROSS,iMIU,icost1,i)
            iMCABATE[i] = fMCABATE(iMIU,i)
            iCPRICE[i] = fCPRICE(iMIU,i)
            iYNET[i] = fYNET(iYGROSS, iDAMFRAC, i)
            iY[i] = fY(iYNET,iABATECOST,i)
            iI[i] = fI(iS,iY,i)
            iC[i] = fC(iY,iI,i)
            iB[i]=fB(iTATM, i)
            iES[i]=fES(iB, i)
            iO[i]=fO(iO,iTATM,i)
            iEQ[i]=fEQ(iES,iO, i)
            iCPC[i] = fCPC(iC,il,i)
            iPERIODU[i] = fPERIODU(iC,iEQ,il,i)
            iCEMUTOTPER[i] = fCEMUTOTPER(iPERIODU,il,i)
            iRI[i] = fRI(iCPC,i)
            
        resUtility = np.zeros(1)
        fUTILITY(iCEMUTOTPER, resUtility)
        
        return (resUtility[0],iI,iK,ial,il,iYGROSS,isigma,iEIND,iE,iCCA,iCCATOT,icumetree,iMAT,iMU,iML,iFORC,iTATM,iTOCEAN,iDAMFRAC,iDAMAGES,iABATECOST,icost1,iMCABATE,
             iCPRICE,iYNET,iY,iC,iB,iES,iO,iEQ,iCPC,iPERIODU,iCEMUTOTPER,iRI)
            
    if __name__ == '__main__':
        
        start = time.time()
        
        TT = np.linspace(2000, 2500, 100, dtype = np.int32)
        
        InitializeLabor(l,NT)
        InitializeTFP(al,NT)
        InitializeGrowthSigma(gsig,NT)
        InitializeSigma(sigma,gsig,cost1,NT)
        InitializeCarbonTree(cumetree,NT)
        x_start = np.concatenate([MIU_start,S_start])
        bnds = bnds1 + bnds2
        
        print('pi={}'.format(pi))
        print('bex={}'.format(bex))
        print('psi={}'.format(psi))

        result = opt.minimize(fOBJ, x_start, args=(-1.0,I,K,al,l,YGROSS,sigma,EIND,E,CCA,CCATOT,cumetree,MAT,MU,ML,FORC,TATM,TOCEAN,DAMFRAC,DAMAGES,ABATECOST,cost1,MCABATE,
             CPRICE,YNET,Y,C,B,ES,O,EQ,CPC,PERIODU,CEMUTOTPER,RI,NT), method='SLSQP',bounds = tuple(bnds),options={'disp': True, 'maxiter':1000})
        FOptimal = Optimality(result.x,I,K,al,l,YGROSS,sigma,EIND,E,CCA,CCATOT,cumetree,MAT,MU,ML,FORC,TATM,TOCEAN,DAMFRAC,DAMAGES,ABATECOST,cost1,MCABATE,
             CPRICE,YNET,Y,C,B,ES,O,EQ,CPC,PERIODU,CEMUTOTPER,RI,NT)
        
        #PlotFigures()
        end = time.time()
        #MainFigures()
        end - start
    
    #Data=pd.DataFrame({'B':FOptimal[-7],'O':FOptimal[-6],'EQ':FOptimal[-6],'TATM':FOptimal[-18],'E':FOptimal[8],'PERIODU':FOptimal[-3]})
    
    end = time.time()
    tot_time+=round((end - start),2)
    #data["df_{}".format(bex)] = pd.DataFrame({'B':FOptimal[-8],'O':FOptimal[-6],'EQ':FOptimal[-5],'TATM':FOptimal[16],'E':FOptimal[8],'PERIODU':FOptimal[-3]})
    
    beta_star=beta*((phi*ES**pi+(1-phi)*O**pi)**(1/pi))**zeta/(beta*(((phi*ES**pi+(1-phi)*O**pi))**(1/pi))**zeta+(1-beta)*(C)**zeta)
    phi_star=phi*ES**pi/(phi*ES**pi+(1-phi)*O**pi)
    data["df_{}".format(pi)] = pd.DataFrame({'B':FOptimal[-8],'O':FOptimal[-6],'ES':FOptimal[-7],'EQ':FOptimal[-5],'TATM':FOptimal[16],'E':FOptimal[8],'EIND':FOptimal[7],'C':FOptimal[-9], 'beta_star':beta_star, 'phi_star':phi_star, 'MAT':MAT})

print('total time:{}'.format(tot_time))




plt.figure()
plt.title('Value Share of NMG in U', fontsize=15)
for pi in pi_list:
    plt.plot(TT[:], 100*data['df_{}'.format(pi)]['beta_star'][:], label='\u03B8 = {}'.format(pi))
    print(pi)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Percent %')
plt.legend(bbox_to_anchor=(0.65, -0.1), ncol=3, fancybox=True, shadow=True)


'''
plt.figure()
plt.ylim(20,80)
plt.title('Value Share of ES in NMG for \u03C0 = -2.5', fontsize=15)
plt.plot(TT[:21], 100*data['df_-2.5']['phi_star'][:21])
plt.fill_between(TT[:21],100*data['df_-2.5']['phi_star'][:21],100, alpha=0.3,label='U(O)')
plt.fill_between(TT[:21],0,100*data['df_-2.5']['phi_star'][:21], alpha=0.3,label='U(ES)')
plt.xlabel('Years', fontsize=12)
plt.ylabel('Percent %')
plt.legend(bbox_to_anchor=(0.65, -0.1), ncol=3, fancybox=True, shadow=True)

'''
plt.figure()
plt.title('Value Share of ES in NMG for \u03C0 = 1', fontsize=15)
plt.ylim(20,80)
plt.plot(TT[:21], 100*data['df_1']['phi_star'][:21])
plt.fill_between(TT[:21],100*data['df_1']['phi_star'][:21],100, alpha=0.3,label='U(O)')
plt.fill_between(TT[:21],0,100*data['df_1']['phi_star'][:21], alpha=0.3,label='U(ES)')
plt.xlabel('Years', fontsize=12)
plt.ylabel('Percent %')
plt.legend(bbox_to_anchor=(0.65, -0.1), ncol=3, fancybox=True, shadow=True)



plt.figure()
plt.title('Value Share of ES in NMG', fontsize=15)
for pi in pi_list:
    plt.plot(TT[:], 100*data['df_{}'.format(pi)]['phi_star'][:], label='\u03B8 = {}'.format(pi))
    print(pi)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Percent %')
plt.legend(bbox_to_anchor=(0.65, -0.1), ncol=3, fancybox=True, shadow=True)




plt.figure()
plt.title('Value Share of NMG in Utility', fontsize=15)
for pi in pi_list:
    plt.plot(TT[:], 100*data['df_{}'.format(pi)]['beta_star'][:], label='\u03B8 = {}'.format(pi))
    print(pi)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Percent %')
plt.legend(bbox_to_anchor=(0.65, -0.1), ncol=3, fancybox=True, shadow=True)



data_DICE=pd.read_csv('DICE_Plots.csv')
data_DICE_dam=pd.read_csv('DICE_Plots_dam.csv')
data_STERN_dam=pd.read_csv('STERN_Plots_dam.csv')
data_RPE=pd.read_csv('DICE_Rel_Plots.csv')
data_STERN=pd.read_csv('STERN_Plots.csv')

data_Thomas=pd.read_csv('DICE_Thomas.csv')
data_2904=pd.read_csv('DICE_Plots_2904.csv')



B_Rel=1-(teta+k*data_RPE['TATM']+fi*data_RPE['TATM']**2)
B_Nord=1-(teta+k*data_DICE['TATM']+fi*data_DICE['TATM']**2)
B_Stern=1-(teta+k*data_STERN['TATM']+fi*data_STERN['TATM']**2)
B_Thomas=1-(teta+k*data_Thomas['TATM']+fi*data_Thomas['TATM']**2)

"""
B_Rel=b0-b0*(0.008432*data_RPE['TATM']**2 - 0.005579 *data_RPE['TATM'] + 0.02761)
B_Nord=b0-b0*(0.008432*data_DICE['TATM']**2 - 0.005579 *data_DICE['TATM'] + 0.02761)
B_Stern=b0-b0*(0.008432*data_STERN['TATM']**2 - 0.005579 *data_DICE['TATM'] + 0.02761)
"""
plt.style.use('seaborn')
sns.set_palette('Set1')

def Plots():
            
        plt.figure()
        plt.title('Optimal Atmospheric Temperature Increase'.format(phi), fontsize=15)
        
        for i in range(len(pi_list)):
            plt.plot(TT[4:21],data['df_{}'.format(pi_list[i])]['TATM'][4:21], label='\u03C0 = {}'.format(pi_list[i]))           
        #plt.plot(TT[4:21], data_DICE['TATM'][4:21], label='Nordhaus')
        plt.plot(TT[4:21], data_DICE_dam['TATM'][4:21], label='Nordhaus')
        plt.plot(TT[4:21], data_RPE['TATM'][4:21], 'g:',label='Drupp & Hänsel', color='black')
        #plt.plot(TT[4:21], data_STERN['TATM'][4:21], label='Stern')
        plt.plot(TT[4:21], data_STERN_dam['TATM'][4:21], label='Stern')
        #plt.plot(TT[4:20], data_AMPL['TATM'][4:20], label='AMPL')
        #plt.plot(TT[4:20], data_AMPL1['TATM'][4:20], label='AMPL1')
        #plt.fill_between(TT[4:20],data['df_{}'.format(min(pi_list))]['TATM'][4:20],data['df_{}'.format(max(pi_list))]['TATM'][4:20],alpha=0.3, color='red')
        
        plt.xlabel('Years', fontsize=12)
        plt.ylabel('°C from 1900', fontsize=12)
        plt.legend( bbox_to_anchor=(0, 1), ncol=1, fancybox=True, shadow=True, fontsize=12)
        
        plt.figure()
        plt.title('Optimal Industrial Emissions ', fontsize=15)
        
        for i in range(len(pi_list)):
            plt.plot(TT[4:21],data['df_{}'.format(pi_list[i])]['EIND'][4:21], label='\u03C0 = {}'.format(pi_list[i]))        
        
        #plt.plot(TT[4:21], data_DICE['EIND'][4:21], label='Nordhaus')
        plt.plot(TT[4:21], data_DICE_dam['EIND'][4:21], label='Nordhaus')
        plt.plot(TT[4:21], data_RPE['EIND'][4:21], 'g:', label='Drupp & Hänsel',color='black')
        #plt.plot(TT[4:21], data_STERN['EIND'][4:21], label='Stern')
        plt.plot(TT[4:21], data_STERN_dam['EIND'][4:21], label='Stern')
        #plt.plot(TT[4:21], data_2904['EIND'][4:21], label='data_2904')
        #plt.fill_between(TT[4:20],data['df_{}'.format(min(pi_list))]['E'][4:20],data['df_{}'.format(max(pi_list))]['E'][4:20],alpha=0.3, color='blue')
        #plt.plot(TT[4:20], data_AMPL['E'][4:20], label='AMPL')
        #plt.plot(TT[4:20], data_AMPL1['E'][4:20], label='AMPL1')
        plt.xlabel('Years', fontsize=12)
        plt.ylabel('GtCO2 per year')

        plt.legend(bbox_to_anchor=(1, -0.1), ncol=5, fancybox=True, shadow=True,fontsize=12)
        
        plt.figure()
        plt.title('Concentration'.format(phi), fontsize=15)
        
    
Plots()

def G(x):    
        result=[]
        for i in range((len(x))-1):
            if i==0:
                result.append(0)
            else:
                result.append((x[i+1]-x[i])/x[i+1])
        return np.array(result)

def R(x):
    result=[]
    for i in range(len(x)):
        result.append(((x[i]+1)**(1/5)-1)*100)
    return result

R(((1-zeta)*(G(C)-G(EQ)*(EQ[:-1]/(EQ[:-1]-EQbar)))))
import seaborn as sns

plt.figure()
fig, axs = plt.subplots(2, 2)
plt.subplots_adjust( hspace=0.4, wspace=0.3,top=1)
axs[0, 0].set_title('Panel A. Growth Rate All-Other Non-Market Goods (O)', fontsize=10)

for i in range(len(pi_list)):
    O=data["df_{}".format(pi_list[i])]['O']
    axs[0, 0].plot(TT[4:20], 100*G(O)[4:20],label=' \u03C0 = {}'.format((pi_list[i])))
    axs[0, 0].set_xlabel('Years')
    axs[0, 0].set_ylabel('Percent / Year')

axs[0, 1].set_title('Panel B. Growth Rate Ecosystem Services (ES) ', fontsize=10)



for i in range(len(pi_list)):
    ES=data["df_{}".format(pi_list[i])]['ES']
    axs[0, 1].plot(TT[4:20], 100*G(ES)[4:20])
    axs[0, 1].set_xlabel('Years')
    axs[0, 1].set_ylabel('Percent / Year')
axs[1, 1].set_title('Panel C.Growth Rate Market Goods (C)', fontsize=10)


for i in range(len(pi_list)):
    C=data["df_{}".format(pi_list[i])]['C']
    axs[1, 1].plot(TT[4:20], 100*G(C)[4:20])
    axs[1, 1].set_xlabel('Years')
    axs[1, 1].set_ylabel('Percent / Year')
axs[1, 0].set_title('Panel D. Growth Rate Non-Market Goods (EQ)', fontsize=10)


for i in range(len(pi_list)):
    EQ=data["df_{}".format(pi_list[i])]['EQ']
    axs[1, 0].plot(TT[4:20], 100*G(EQ)[4:20])
    axs[1, 0].set_xlabel('Years')
    axs[1, 0].set_ylabel('Percent / Year')
labels=['\u03C0 = -1','\u03C0 = 1']
fig.legend(labels=labels,loc='lower center' ,ncol=2)



plt.figure()
plt.title('Relative Price Effect of Non-Market Goods')
for i in range(len(pi_list)):
   C=data["df_{}".format(pi_list[i])]['C']
   EQ=data["df_{}".format(pi_list[i])]['EQ']
   
   sns.lineplot(TT[4:21],R(((1-zeta)*(G(C)-G(EQ)*(EQ[:-1]/(EQ[:-1]-EQbar)))))[4:21], label=' \u03C0 = {}'.format((pi_list[i])))
   
   print(R(((1-zeta)*(G(C)-G(EQ)*(EQ[:-1]/(EQ[:-1]-EQbar)))))[4])
   plt.xlabel('Years', fontsize=12)
   plt.ylabel('Percent / Year', fontsize=12)
   plt.legend()


def Plots_RPE():
    
    plt.figure()
    plt.title('Relative Price Effect O-ES', fontsize=18)
    for i in range(len(pi_list)):
        ES=data["df_{}".format(pi_list[i])]['ES']
        O=data["df_{}".format(pi_list[i])]['O']
        RPE=R((1-pi_list[i])*(G(ES)-G(O)))
        plt.xlim([2020,2100])
        plt.plot(TT[4:21],RPE[4:21], label='\u03C0= {}'.format(pi_list[i]))
        plt.xlabel('Years', fontsize=16)
        plt.ylabel('Percent / Year', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0.75, -0.15), ncol=2, fancybox=True, shadow=True,fontsize=16)
        

    plt.figure()
    plt.title('Relative Price Effect O-ES 500 years')
    for i in range(len(pi_list)):
        ES=data["df_{}".format(pi_list[i])]['ES']
        O=data["df_{}".format(pi_list[i])]['O']
        RPE=100*(pi_list[i]-1)*(G(ES)-G(O)) 
        plt.plot(TT[4:-1],RPE[4:], label='RPE \u03C0= {}'.format(pi_list[i]))
        plt.xlabel('Years', fontsize=12)
        plt.ylabel('Percent / Year', fontsize=12)
        plt.legend()
    
    plt.figure()
    plt.title('Relative Price Effect EQ-C 100 years')
    for i in range(len(pi_list)):
       C=data["df_{}".format(pi_list[i])]['C']
       EQ=data["df_{}".format(pi_list[i])]['EQ']
       RPE=100*((1-zeta)*(G(C)-G(EQ)*(EQ[:-1]/(EQ[:-1]-EQbar))))
       print(RPE[4])
       
       
       plt.xlabel('Years', fontsize=12)
       plt.ylabel('Percent / Year', fontsize=12)
       plt.legend()  
 
    plt.figure()
    plt.title('ES')
    for i in range(len(pi_list)):
        ES=data["df_{}".format(pi_list[i])]['ES']
        plt.plot(TT[4:20],(ES)[4:20], label=' pi = {}'.format(pi_list[i]))
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Percent / Year', fontsize=12)
    plt.legend(loc='lower right')
    
Plots_RPE()


B_Rel=(teta+k*(data_RPE['TATM'])+fi*(data_RPE['TATM'])**2)
B_Nord=(teta+k*(data_DICE['TATM'])+fi*(data_DICE['TATM'])**2)
B_Stern=(teta+k*(data_STERN['TATM'])+fi*(data_STERN['TATM'])**2)

#B_Nord_t=(teta_t+k_t*(data_DICE['TATM']-data_DICE['TATM'][4])+fi_t*(data_DICE['TATM']-data_DICE['TATM'][4])**2)
#B_Stern_t=(teta_t+k_t*(data_STERN['TATM']-data_STERN['TATM'][4])+fi_t*(data_STERN['TATM']-data_STERN['TATM'][4])**2)
       

plt.figure()
plt.title('Projected Climate Change-driven Biodiversity Loss', fontsize=18)         
plt.plot(TT[4:21], data['df_-2.5']['B'][4:21]*100, label='B-DICE')
#plt.plot(TT[4:21], B_Stern_t[4:21] ,color='tab:red', label='\u03B4 = 0.1%')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.plot(TT[4:21], B_Rel[4:21], label=('Drupp & Hänsel'))
#plt.plot(TT[4:21], B_Nord_t[4:21],  color='tab:blue')
plt.plot(TT[4:21], B_Nord[4:21]*100,  label='Nordhaus')

plt.plot(TT[4:21], B_Stern[4:21]*100,  label='Stern')
#plt.fill_between(TT[4:21],B_Nord[4:21],B_Stern[4:21],alpha=0.3, color='tab:blue', label='Urban (2015)')
#plt.fill_between(TT[4:21],B_Nord_t[4:21],B_Stern_t[4:21],alpha=0.3, color='tab:red', label='Thomas et al. (2004)')
#plt.fill_between(TT[4:21],B_Stern_t[4:21],B_Nord[4:21],alpha=0.25, color='tab:grey')
plt.xlabel('Years', fontsize=14)
plt.ylabel('% of Species Extinction', fontsize=14)
plt.legend()
plt.legend( bbox_to_anchor=((0.77, -0.13)), ncol=3, fancybox=True, shadow=True, fontsize=12)

for i in range(len(data['df_-2.5'])-1):
    r=(data['df_-2.5']['B'][i+1]-data['df_-2.5']['B'][i])*100
    print(r)
    
DICE_MAT=pd.read_csv('DICE_MAT.csv')
standard_DICE_MAT=pd.read_csv('standardDICE_MAT.csv')
STERN_MAT=pd.read_csv('STERN_MAT.csv')

text_kwargs = dict(ha='center', va='center', fontsize=28, color='C1')
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

fig=plt.figure()


plt.title('Projected $CO_2$ Concentrations in the Atmosphere', fontsize=18)
plt.ylim([300,650])
for i in range(len(pi_list)-1):
    plt.plot(TT[4:21],data['df_{}'.format(pi_list[i])]['MAT'][4:21]/2.13, label='B-DICE')           
    #plt.plot(TT[4:21], data_DICE['MAT'][4:21], label='Nordhaus')
plt.plot(TT[4:21], DICE_MAT['MAT'][4:21]/2.13, label='Nordhaus')
#plt.plot(TT[4:21], standard_DICE_MAT['MAT'][4:21]/2.13, label='standard Nordhaus')
plt.xlabel('Years', fontsize=14)
plt.ylabel('Parts per million (ppm)', fontsize=14)
plt.plot(TT[4:21], STERN_MAT['MAT'][4:21]/2.13, label='Stern')
plt.fill_between(TT[4:21],0,350,alpha=0.3, color='green')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.fill_between(TT[4:21],350,550,alpha=0.3, color='yellow')
plt.fill_between(TT[4:21],550,650, alpha=0.3, color='orange')
plt.legend( bbox_to_anchor=((0.825, -0.13)), ncol=3, fancybox=True, shadow=True, fontsize=14)
plt.show()

def R(x):
    result=[]
    for i in range(len(x)):
        result.append(((x[i]+1)**(1/5)-1)*100)
    return result

def rel(x):
    r=[]
    for i in range(len(x)-1):
        r.append((((x[i+1]-x[i])/5)*100000))
    return r

B=data['df_-2.5']['B']


rel(B)

plt.figure()
plt.title('Projected Climate Change-driven Extinction Rates', fontsize=18)
plt.ylim([0,150])
plt.plot(TT[4:21],rel(B)[4:21], label='B-DICE')
plt.plot(TT[4:21], rel(B_Nord)[4:21], label='Nordhaus')
plt.plot(TT[4:21], rel(B_Stern)[4:21], label='Stern')
plt.xlabel('Years', fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Average extinction per million species per year', fontsize=14)
plt.fill_between(TT[4:21],10,150, alpha=0.3, color='orange')
plt.fill_between(TT[4:21],0,10, alpha=0.3, color='green')
plt.legend( bbox_to_anchor=((0.825, -0.13)), ncol=3, fancybox=True, shadow=True, fontsize=14)

