#!/usr/bin/env python

# Programmer(s): Vincent Adombi

""" MAIN PROGRAM FILE
Run this file to perform a single run of the EXP-HYDRO model
with user provided parameter values.

Inspired from Sopan Patil
"""

import numpy
import os
import matplotlib.pyplot as plt
from conceptual_models.lumped import ExphydroModel, ExphydroParameters, ABCDModel, ABCDParameters
from hydroutils import ObjectiveFunction

######################################################################
# SET WORKING DIRECTORY

# Getting current directory, i.e., directory containing this file
dir1 = os.path.dirname(os.path.abspath('__file__'))

# Setting to current directory
os.chdir(dir1)

######################################################################
# MAIN PROGRAM

# Load meteorological and observed flow data
P = numpy.genfromtxt('SampleData/P_test.txt')  # Observed rainfall (mm/day)
T = numpy.genfromtxt('SampleData/T_test.txt')  # Observed air temperature (deg C)
PET = numpy.genfromtxt('SampleData/PET_test.txt')  # Potential evapotranspiration (mm/day)
Qobs = numpy.genfromtxt('SampleData/Q_test.txt')  # Observed streamflow (mm/day)

# Initialise EXP-HYDRO model parameters object
params = ABCDParameters()

# Specify the parameter values
# Please refer to Patil and Stieglitz (2014) for model parameter descriptions
ddf = 2
mint = -1
maxt = 1
a = 0.5
b = 100.0
c = 0.5
d = 0.5

# Assign the above parameter values into the model parameters object
params.assignvalues(ddf, mint, maxt, a, b, c, d)

# Initialise the model by loading its climate inputs
model = ABCDModel(P, PET, T)

# Specify the start and end day numbers of the simulation period.
# This is done separately for the observed and simulated data
# because they might not be of the same length in some cases.
simperiods_obs = [365, 2557]
simperiods_sim = [365, 2557]

# Run the model and calculate objective function value for the simulation period
Qsim = model.simulate(params)
kge = ObjectiveFunction.klinggupta(Qobs[simperiods_obs[0]:simperiods_obs[1]+1],
                                   Qsim[simperiods_sim[0]:simperiods_sim[1]+1])
print('KGE value = ', kge)

# Plot the observed and simulated hydrographs
plt.plot(Qobs[simperiods_obs[0]:simperiods_obs[1]+1], 'b-')
plt.plot(Qsim[simperiods_sim[0]:simperiods_sim[1]+1], 'r-')
plt.show()

######################################################################
