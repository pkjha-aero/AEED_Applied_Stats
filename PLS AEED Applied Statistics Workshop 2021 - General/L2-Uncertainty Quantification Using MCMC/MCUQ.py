#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:34:06 2021
MC uncertainty propagation library - daq

@author: dquint
"""

import numpy as np
from scipy import optimize, stats, integrate

# RANGE EQ. - Used to generate fake experimental data
def modelFit(alpha, g):
    
    v0=1.
    r = (v0 ** 2) * np.sin(2.*alpha)/g
    
    return r

# 2D PROJ. MOTION THEORY
def Thry(time, initial, alpha, g):
    
    x0, y0, v0 = initial 
    
    x = x0 + v0*np.cos(alpha)*time
    y = y0 + v0*np.sin(alpha)*time - 0.5*g* np.power(time,2.)
    
    return x,y

# COARSE GRAIN SIMULATION USING FANCY NUMPY INTEGRATORS
def Sim(time, initial, g, alpha, ncg):
    
    if ncg < len(time):
        cgtime = np.linspace(time[0],time[len(time)-1],ncg)
    else:
        cgtime = time
        
    x0, y0, v0 = initial
    
    v0x = np.cos(alpha) * v0
    v0y = np.sin(alpha) * v0
    
    vx = v0x*np.ones(len(cgtime))
    vy = v0y - 1.0*g*cgtime
    
    x = integrate.cumtrapz(vx, cgtime)
    y = integrate.cumtrapz(vy, cgtime)
    
    y = np.insert(y,0,0.0)
    x = np.insert(x,0,0.0)
    
    return x,y, cgtime

# ITERATIVELY RUN THE OBJECTIVES
def runSim(alphaMeas, time, initial, ncg, g):
    
    xranges = []
    
    for alpha in alphaMeas:
        x,y,cgtime = Sim(time, initial, g, alpha, ncg)
        tidx, idx, xdisp, ydisp = Range(x,y,cgtime)
        xranges.append(xdisp)
    
    return np.asarray(xranges)

def runModel(alphaMeas, g):
    
    xranges = []
    
    for alpha in alphaMeas:
        
        xr = modelFit(alpha,g)
        xranges.append(xr)
    
    return np.asarray(xranges)

def runTheory(alphaMeas, time, initial, g):
    
    xranges = []
    
    for alpha in alphaMeas:
        x,y = Thry(time, initial, alpha, g)
        tidx, idx, xdisp, ydisp = Range(x,y,time)
        xranges.append(xdisp)
    return np.asarray(xranges)

# DETERMINE THE RANGE TRAVELED 
def Range(x, y, time):
    
    # The result here will depend on the coarse-ness of the time steps
    # used in the simulation/theory
    
    tidx = y >= 0.
    totTime = time[tidx].max()
    idx = (np.abs(time - totTime)).argmin()
    xdisp = x[idx]
    ydisp = y[idx]
    
    return tidx, idx, xdisp, ydisp

## BAYES MCMC POSTERIOR DEFINITIONS ##
def logprior(g):
    
    if 9.0 < g < 20.0:
        return 0.0
    else:
        return -np.infty

def loglikelihood(rObs, rSim, rErr):
    
    return -0.5 * np.sum(np.log(2 * np.pi * rErr ** 2)
                         + (rObs - rSim) ** 2 / rErr ** 2)

def theorylogposterior(g, model, time, alphaMeas, initial, rObs, rErr):
    
    #initial = x0,y0,v0
    #model, time, alphaMeas, initial, yMeas, yErr = args
    
    rMod = model(alphaMeas, time, initial, g)
    
    lh = loglikelihood(rObs, rMod, rErr)
   
    lp = logprior(g)
    
    return lp+lh if np.isfinite(lp+lh) else -np.infty


def modellogposterior(g, model, alphaMeas, rObs, rErr):
    
    rMod = model(alphaMeas, g)
    
    lh = loglikelihood(rObs, rMod, rErr)
   
    lp = logprior(g)
    
    return lp+lh if np.isfinite(lp+lh) else -np.infty

def simlogposterior(g, model, time, alphaMeas, initial, ncg, rObs, rErr):
    
    #initial = [x0,y0,v0]
    #model, time, alphaMeas, initial, yMeas, yErr = args
    
    rSim = model(alphaMeas, time, initial, ncg, g)
    
    lh = loglikelihood(rObs, rSim, rErr)
   
    lp = logprior(g)
    
    return lp+lh if np.isfinite(lp+lh) else -np.infty

