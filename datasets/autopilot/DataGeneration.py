import numpy as np
import pandas as pd
import os
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch
import csv
from collections.abc import Iterable
from _operator import truediv


# path1 = pathlib.Path.cwd()
# print(path1)
# path2 = os.getcwd()
# print(path2)

class GenerateData:
    def __init__(self) ->None:
        pass

    def get_data(self, dataDir='dataverse_files/patrol_ship_routine/processed/', type='train', exp_time='20190805-095929', parameter_exp=['u', 'deltal', 'r'], 
                 transform = None, seqLength = None):
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
            input_para = data[:,:-1]
            output_para = data[:,-1]
            #print(input.shape, output.shape)
            input_list = input_para.numpy().tolist() 
            output_list = output_para.numpy().tolist()
            print(len(input_list),len(output_list))        
        else:
            print('no parameter')


        if seqLength is not None:
            temp = [] 
            temp2 = []
            for i in range(round(truediv(len(input_list),seqLength))):
                if (len(input_list) - i*seqLength) < seqLength:
                    pass 
                     #right = len(t)

                else: 
                    right = (i+1)*seqLength
                    temp.append(input_list[i*seqLength:right])
                    temp2.append(output_list[i*seqLength:right])
                                
            input = temp
            output = temp2
            print(input, len(input), output, len(output))
            input_tensor = torch.tensor(input)
            output_tensor = torch.tensor(output)
            print(input_tensor.shape, output_tensor.shape)

            torch_dataset = Data.TensorDataset(input_tensor, output_tensor)
            
        else: 
            input = input_list
            output = output_list
             
        return torch_dataset
    

TrainData = GenerateData().get_data(dataDir='dataverse_files/patrol_ship_routine/processed/', type='train', exp_time='20190805-095929', parameter_exp=['u', 'deltal', 'r'], seqLength = 40)
train_loader = torch.utils.data.DataLoader(dataset=TrainData,batch_size=16,shuffle=False)

for i, (data, target) in enumerate(train_loader):
    print(data.shape,target.shape)




#print(isinstance(['u', 'deltal', 'r'], Iterable))



    
    




