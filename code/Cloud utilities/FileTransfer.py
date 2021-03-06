# -*- coding: utf-8 -*-
"""
Created on Oct 4, 2020

@author: ahinoamp@gmail.com

This script transfers a random selection of history files + the best score 
history file into a separate folder
"""
import os
import re
import random   
import numpy as np
from shutil import copyfile

#Loop through all of the files and take the best and and another 5 random ones and copy them over
def GetErrFromFile(file):
    result = re.search('Err_(.*).his', file)
    return float(result.group(1))

folder= 'Combo_Scratch/'

folders = [os.path.isdir(folder+'/'+i) for i in os.listdir(folder)]

from glob import glob
folders = glob("Combo_Scratch/*/")
nFolders = len(folders)

MasterFileList = []
for i in range(nFolders):
    threadfolder = folders[i]
    print('reading: '+ threadfolder)
    history_file_folder =threadfolder+'HistoryFileInspection/'
    historyfiles = glob(history_file_folder+"*.his")
    ErrList = []
    for h in range(len(historyfiles)):
        hisfile = historyfiles[h]
        err = GetErrFromFile(hisfile)
        ErrList.append(err)
    
    if(len(historyfiles)<1):
        continue

    kval = np.min([len(historyfiles), 1])
        
    files2copy = list(random.choices(historyfiles, k=kval))
    bestfile = np.argmin(ErrList)
    files2copy.append(historyfiles[bestfile])
    MasterFileList = MasterFileList+files2copy

print(len(MasterFileList))
folderPreTransfer= 'HistoryFileTransfer/'
for i in range(len(MasterFileList)):
    print(i)
    shortenName = re.search('His_(.*)', MasterFileList[i]).group(1)
    copyfile(MasterFileList[i], folderPreTransfer+shortenName) 
