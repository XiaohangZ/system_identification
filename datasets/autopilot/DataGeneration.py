import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pathlib
import torch
import csv
from collections.abc import Iterable


# path1 = pathlib.Path.cwd()
# print(path1)
# path2 = os.getcwd()
# print(path2)

class GenerateData(torch.utils.data.Dataset):
    def __init__(self) ->None:
        pass

    def get_data(self, dataDir='dataverse_files/patrol_ship_routine/processed/', type='train', exp_time='20190805-095929', parameter_exp=['u', 'deltal', 'r']):
        print(dataDir+type+'/'+exp_time+".csv")

        csvFiles = dataDir+type+'/'+exp_time+".csv"
        
        # with open(csvFiles, encoding='utf-8-sig') as file:
        #     row = csv.reader(file)
        #     for r in row:
        #         print(r)

        data_numpy = np.loadtxt(csvFiles, delimiter=",", skiprows=1)
        parameter_list = next(csv.reader(open(csvFiles), delimiter=','))
        #print(data_numpy.shape, parameter_list)
        data = torch.from_numpy(data_numpy)

        if parameter_exp is not None:
            data_exp_list=[]
            for parameter_name in parameter_exp:
                if parameter_name in parameter_list:
                    #print(parameter_list.index(parameter_name), parameter_name)
                    exp_index=torch.tensor(parameter_list.index(parameter_name))
                    #print (exp_index)
                    data_exp=torch.index_select(data, 1, exp_index)
                    para_exp=parameter_list[exp_index.int()]
                    #print(para_exp, data_exp)
                data_exp_list.append(data_exp)
                #print(data_exp_list)
            data=torch.cat(data_exp_list,dim=1)
            print(data.shape)
            input = data[:,:-1]
            output = data[:,-1]
            print(input.shape, output.shape)        
        else:
            print('no parameter')
        
            
        return input,output

print(GenerateData().get_data())
#print(isinstance(['u', 'deltal', 'r'], Iterable))



    
    




