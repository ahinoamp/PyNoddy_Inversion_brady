# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:24:56 2020

@author: ahinoamp@gmail.com

This script sends out the threads to the respective algorithms listed in the 
hyperparameter dictionary
"""

import MCMC_Noddy as mcmc
import GA_Noddy as GA
import PSO_basic as PSO
import NSGA_Noddy as NSGA
from pathlib import Path

def OptimisationSwitchBoard(P):
    
    if(P['OptimMethod']=='MCMC'):
        mcmc.MCMC_Noddy(P)        
    elif(P['OptimMethod']=='GA'):
        GA.GA_Noddy(P)            
    elif(P['OptimMethod']=='Annealing'):
        mcmc.MCMC_Noddy(P)                    
    elif(P['OptimMethod']=='NSGA'):
        NSGA.NSGA2_Noddy(P)                            

    folder = P['BaseFolder']+'/Thread'+str(P['thread_num'])+'/'
    for p in Path(folder).glob("noddy_out*"):
        p.unlink()        
        
        