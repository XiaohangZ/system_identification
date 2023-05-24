import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pathlib
import torch
import csv


#path1 = pathlib.Path.cwd()
#print(path1)
#path2 = os.getcwd()
#print(path2)

class GenerateData(torch.utils.data.Dataset):
    def __init__(self, dataDir='dataverse_files/patrol_ship_routine/processed/', type='train', exp_time='20190805-095929'):
        print(dataDir+type+'/'+exp_time+".csv")

        csvFiles = dataDir+type+'/'+exp_time+".csv"
        
        with open(csvFiles) as file:
            row = csv.reader(file)
            for r in row:
                print(r)

        return

GenerateData()



    
    




