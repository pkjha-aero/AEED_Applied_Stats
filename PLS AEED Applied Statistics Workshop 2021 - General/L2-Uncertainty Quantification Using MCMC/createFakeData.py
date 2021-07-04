#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:14:50 2021
code for creating fake experiment data 
@author: quint2
"""

from MCUQ import * # This is a set of custom functions we will use
import pandas as pd

rng = np.random.RandomState(314) # use the same seed for reproducibility 
alphaMeas = np.linspace(np.pi/20, np.pi/2.1, 20)
time = np.linspace(0,1, 1000)
x0, y0, v0 = 0., 0., 1.
initial =[x0,y0,v0]
g = 9.8
rErr = 0.005 #error magnitude

rTrue = runTheory(alphaMeas,time,initial,g)
rObs = np.asarray(rTrue) + rng.normal(scale=rErr, size = np.size(rTrue))

alphaMeas = np.reshape(alphaMeas,(len(alphaMeas),1))
rObs = np.reshape(rObs,(len(rObs),1))
rErr = 0.005*np.ones((len(rObs),1))

data = np.hstack((alphaMeas,rObs,rErr))

df = pd.DataFrame(data=data, columns=['alpha','rObs','rErr'])
df.to_csv('range_data_exp.csv')
# np.savetxt('range_data_exp.csv',rObs,delimiter=",")


dfl = pd.read_csv('range_data_exp.csv')
