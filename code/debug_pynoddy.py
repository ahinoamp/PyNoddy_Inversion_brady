# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 22:21:11 2021

@author: ahino
"""

import sys
import os
import MCMC_Noddy as mcmc
import GA_Noddy as GA
import PSO_basic as PSO
import NSGA_Noddy as NSGA
import glob
import vedo as vtkP
import pandas as pd
import re 

# 3D plotting utilities
import plot3d_util as plt3d
import vedo as vtkP
import pandas as pd
from scipy.spatial import Delaunay

HyperParameters = {}

# what optimisation method are you using?
# choose from ['MCMC', 'GA', 'NSGA', 'Annealing']
HyperParameters['OptimMethod']='GA'

# The number of faults in the model ::: (integer) :::
# If less than 6, then it's a pre configured scenario
HyperParameters['ScenarioNum'] = 12

# how to shift the geophysical data to be similar to 
# simulation values ['Median Datum Shift', 'Median Const Datum Shift']
HyperParameters['SimulationShiftType'] = 'Median Datum Shift'

# how to normalize each data type before combining them ::: ['MedianInitialRounds', 'Given']
HyperParameters['DatNormMethod'] = 'MedianInitialRounds'
# you can also preconfigure what is the beginning error level for each datatype as follows. 
# this is used only if provided "given" in above option
HyperParameters['DatNormCoef'] = {'Grav': 2.4, 'Tracer': 1.0, 
                        'FaultMarkers': 500, 'GT': 315, 'Mag':300}

# In the MCMC and annealing method. Whether to have a multiplier on the 
# stepping size based on the error level. step = (Multiplier)*(error btwn 0-1)*perturbation 
# ::: ['LinearErrorBased', 'None']
HyperParameters['ExplorationRate'] = 'LinearErrorBased'
HyperParameters['SteppingSizeMult'] = 1/0.9

# Is the error calculated based on mismatch for all observed data from the whole model [Global], 
# or only data points that are local/around the location of the parameter/fault [Local]. 
HyperParameters['ErrorType'] = 'Global'

# If you choose "Local", then how often should you update what ares
# is considered local for every parameter (update once or many times)? :::['Once', 'Many']
HyperParameters['parameters']= 'Many'
# The local radius is calculated by taking the fault length, and this radius can be multiplied by a value. ::: [1, 2.5]
HyperParameters['localWeightRadiusMult']= 1.5
# after how many rounds update the area designated as local? and every how many rounds recalculate?
HyperParameters['StartUpdateLocalWeight']= 45
HyperParameters['localWeightRadiusMult']= 35

# The norm used for the optimisation ::: ['L1', 'L2', 'Lhalf']
HyperParameters['ErrorNorm'] = 'L1'

# For MCMC and annealing, should you set the initial rounds as exploratory, 
# without any optimisation/search/update
HyperParameters['ExplorationStage'] = 'Explore'
# The number of exploration rounds ::: integer
HyperParameters['nExploreRuns'] = 50

# What is the normalization constant in the MCMC (std of the error)? 
# It can be set to achieve a certain acceptance rate, for example.
#::: ['Track Acceptance', 'Error must decrease', 'Const diff']
HyperParameters['AcceptProbType'] = 'Track Acceptance'
# If you want to have a target acceptance, that what is the goal? ::: range(0,1)
HyperParameters['AcceptanceGoal'] = 0.2
# If you want a constant number, set it here
HyperParameters['ConstNormFactor'] = 0.01

# cube size for the model
HyperParameters['cubesize'] = 150

# What is the largest amount the faults can move from their original placed location
HyperParameters['GlobalMoveEachDir'] = 700
# What is the std of the step size for this movement?
HyperParameters['XYZ_Axes_StepStd'] = 100
# Std for the step size of the fault dip
HyperParameters['Dip_StepStd'] = 3
# Std for the step size of the fault slip
HyperParameters['Slip_StepStd'] = 70
# Std for the step size of the fault dip direction
HyperParameters['DipDirection_StepStd'] = 7
# The ratio between fault length to fault slip
HyperParameters['SlipParam'] = 0.1

# Maximum range of the stratigraphy rotation
HyperParameters['AzimuthMoveEachDirection'] = 5
# Maximum change in dip
HyperParameters['DipMoveEachDirection'] = 35
# Maximum shrinking/expanding of fault amplitude
HyperParameters['AmplitudeRatioChange'] = 0.15

HyperParameters['AxisRatioChange'] = 0.1

# The maximum misfit error per fault marker (otherwise can be inf)
HyperParameters['MaxFaultMarkerError'] = 525

HyperParameters['MO_WeightingMethod'] = 'Proportions'
HyperParameters['MCMC_SwitchWeightFreq'] = 20

HyperParameters['thread_num'] = 0        
# don't do the toy run
HyperParameters['Toy']=False

# do you want some extra output
HyperParameters['verbose']=True

# where to output results
HyperParameters['BaseFolder']='Combo_Scratch'

HyperParameters['DataTypes'] = ['Grav', 'GT', 'MVT', 'FaultMarkers','Tracer']

# origin of the model
HyperParameters['xy_origin']=[325233.059, 4404112, -2700]

# extent of the model
HyperParameters['xy_extent'] = [4950, 6150, 3900]

# index of the granite layer
HyperParameters['graniteIdx'] = 4

# is this a Windows computer? (you are probably running this using binder on the server, which is linux)
HyperParameters['Windows'] = True

# are you running this from a jupyter notebook? or your terminal
HyperParameters['jupyter'] = True

# Simulated annealing
HyperParameters['AcceptProbType'] = 'Annealing'

# Annealing: what is the initial temperature? ::: [0.001, 0.025]
HyperParameters['InitialTemperature'] = 0.007
# Annealing: what is the ReductionRate? ::: [0.95, 0.999]
HyperParameters['ReductionRate'] = 0.965

# how much weight to give to each data type? (multi objective)
# equal? randomly set the proportions? all weight to single data type, but switch around every few rounds?
# ::: ['Proportions', 'Extreme', 'Equal']
HyperParameters['MO_WeightingMethod'] = 'Equal'
# if "extreme", every how many rounds switch the weights? ::: integer
HyperParameters['MO_SwitchWeightFreq'] = 2
# rounds are shorter for MCMC, so can have more rounds before switching
HyperParameters['MCMC_SwitchWeightFreq'] = 20

# Genetic algorithms parameters (incl. NSGA)
# #########################################

# number of individuals/models in the population ::: [25, 80]
HyperParameters['npop'] = 35
    
# what is the selection method? ::: ['selTournament', 'selStochasticUniversalSampling', 'selRoulette']
HyperParameters['SelectionMethod'] = 'selTournament'
# what is the tournament size? ::: [4,12]
HyperParameters['TournamentSize'] = 5

# what is the mating method? ::: ['cxTwoPoint','cxOnePoint','cxUniform']
HyperParameters['MatingMethodGlobal'] = 'cxTwoPoint'

# what is the mating method for a local formulation? ::: ['cxOnePointLocal','cxTwoPointLocal', 'cxLocalErrorPropExchange']
HyperParameters['MatingMethodLocal'] = 'cxLocalErrorPropExchange'

# parameter used in the mating algorithms ::: [0.6, 1] 
HyperParameters['MatingSwapRange'] = 0.7

# parameter used in the mating algorithm cxUniform ::: [0.3, 0.7]
HyperParameters['MatingSwapProb'] = 0.4

# What is the mating probability ::: [0.6, 1] or [0-1] in general
HyperParameters['IndMatingProb'] = 0.9

# what is the mutating method? ::: ['mutPolynomialBounded', 'mutGaussian', 'mutUniformFloat']
HyperParameters['MutatingMethod'] = 'mutPolynomialBounded'

# parameter used in the algorithm mutPolynomialBounded ::: [80, 120]
HyperParameters['Eta'] = 100

# What is the individual mutating probability ::: [0.2, 0.4] or [0-1] in general
HyperParameters['IndMutatingProb'] = 0.3
HyperParameters['PbMutateParameter'] = 0.25

HyperParameters['LocalWeightsMode'] = 'na'

# for this run
# how many runs (besides the exploration runs if doing MCMC/annealing)
HyperParameters['nruns']=200
# output image every X iterations
HyperParameters['OutputImageFreq'] = 5        
# what optimisation method?
# choose from ['MCMC', 'GA', 'NSGA', 'Annealing']
HyperParameters['OptimMethod']='Annealing'


if(HyperParameters['OptimMethod']=='MCMC'):
    mcmc.MCMC_Noddy(HyperParameters)        
elif(HyperParameters['OptimMethod']=='GA'):
    HyperParameters['ngen']=int(HyperParameters['nruns']/HyperParameters['npop'])
    HyperParameters['OutputImageFreq'] = 1        
    GA.GA_Noddy(HyperParameters)            
elif(HyperParameters['OptimMethod']=='Annealing'):
    mcmc.MCMC_Noddy(HyperParameters)                    
elif(HyperParameters['OptimMethod']=='NSGA'):
    HyperParameters['ngen']=int(HyperParameters['nruns']/HyperParameters['npop'])
    HyperParameters['OutputImageFreq'] = 1        
    NSGA.NSGA2_Noddy(HyperParameters)                            