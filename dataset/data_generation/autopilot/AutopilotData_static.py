from fractions import Fraction
import math
import random
import torch
from _operator import truediv
import torch.utils.data as Data
import numpy as np
import pandas as pd
import sys
import os

o_path = os.getcwd()
sys.path.append(o_path)

# def random_static_input(
#     N: int,
#     nu: int):

#     us = np.zeros(shape=(nu, N))
#     for element in range(nu):
#         k_start = 0
#         k_end = 0
#         while k_end < N:
#             k_end = k_start + int(
#                 np.random.uniform(
#                     low=3, high=5
#                 )
#             )
#             amplitude = np.random.uniform(low=1, high=2)
#             us[element, k_start:k_end] = amplitude
#             k_start = k_end
#     return [np.array(u).reshape(nu, 1) for u in us.T]

# print(random_static_input(100, 1))

class autopilot_dataset:
    #input time duration
    def __init__(self) -> None:
        pass

    def generate_data(self, T, seqLength = None, model = 'Train'):
        #define initial state
        self.model = model
        if model == 'Train':
            U = random.uniform(3, 8)
        if model == 'Validation':
            U = random.uniform(3, 8)
        if model == 'Test':
            U = random.uniform(8, 12)

        delta_max = 30*math.pi/180
        dot_delta_max = 10*math.pi/180
        delta = random.uniform(-delta_max,delta_max)
        #print(U,delta)

        U_list=[]
        delta_list=[]
        r_list=[]
        r_list_sig=[]
        t_list=[]
        t=0
        t_start=0
        t_end=0


        T_dotr = Fraction(221,5)
        T_U_dotr = -Fraction(662,45)
        T_U2_dotr = Fraction(449,180)
        T_U3_dotr = -Fraction(193,1620)
        K_delta = -Fraction(7,100)
        K_U_delta = Fraction(1,360)
        K_U2_delta = Fraction(1,180)
        K_U3_delta = -Fraction(1,3024)
        N_r = 1
        N_r3 = Fraction(1,2)
        N_U_r3 = -Fraction(43,180)
        N_U2_r3 = Fraction(1,18)
        N_U3_r3 = -Fraction(1,324)
        
        while t_end < T:

            r = 0 
            F_rudder = K_delta*delta + K_U_delta*U*delta + K_U2_delta*math.pow(U,2)*delta + K_U3_delta*math.pow(U,3)*delta
            F_hydro = N_r*r + N_r3*math.pow(r,3) + N_U_r3*U*math.pow(r,3) + N_U2_r3*math.pow(U,2)*math.pow(r,3) + N_U3_r3*math.pow(U,3)*math.pow(r,3)
            dot_r = (F_rudder - F_hydro) / (T_dotr + T_U_dotr*U + T_U2_dotr*math.pow(U,2) + T_U3_dotr*math.pow(U,3))
            r = r + 0.01*dot_r
            
            t_end = t_start + int(random.uniform(3,5))

            while t_end - t_start > 0:
                t = t + 1
                t_list.append(t)

                U_list.append(U)
                delta_list.append(delta)
                r_list.append(r)
                r_list_sig.append([r])

                t_start = t_start + 1
            

            dot_delta = random.uniform(-dot_delta_max, dot_delta_max) 
            delta = delta + dot_delta

        U_tensor = torch.tensor(U_list)
        delta_tensor = torch.tensor(delta_list)
        r_tensor = torch.tensor(r_list)
        r_tensor_sig = torch.tensor(r_list_sig)
        data_tensor = torch.stack([U_tensor, delta_tensor, r_tensor], dim=1)
        input_tensor = torch.stack([U_tensor, delta_tensor], dim=1)
        input_list = input_tensor.tolist()
        output_tensor = r_tensor_sig
        output_list = r_tensor_sig.tolist()


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

            input_seq = temp
            output_seq = temp2
            #print(input_seq, len(input_seq), output_seq, len(output_seq))
            input_tensor = torch.tensor(input_seq)
            output_tensor = torch.tensor(output_seq)
            #print(input_tensor.shape, output_tensor.shape)

            torch_dataset = Data.TensorDataset(input_tensor, output_tensor)

        else:
            input_tensor = torch.tensor(input_list)
            output_tensor = torch.tensor(output_list)
            torch_dataset = Data.TensorDataset(input_tensor, output_tensor)
        

        data_list = data_tensor.tolist()
        df1 = pd.DataFrame(data = data_list,
        columns=['U', 'delta', 'r'])
        if model == 'Train':
            df1.to_csv('dataset/DATASET_DIRECTORY/processed/train/train_10.csv',index=False)
        if model == 'Validation':
            df1.to_csv('dataset/DATASET_DIRECTORY/processed/validation/validation_10.csv',index=False)
        if model == 'Test':
            df1.to_csv('dataset/DATASET_DIRECTORY/processed/test/test.csv',index=False)
                 
        return torch_dataset

    def generate_train_data(self, model = 'train'):
        data_dir = "dataset/DATASET_DIRECTORY/processed/" + model
        filenames = list()
        filenames += [model + '_2.csv', model + '_3.csv', model + '_4.csv']
        filenames += [model + '_5.csv', model + '_6.csv', model + '_7.csv', model + '_8.csv']
        filenames += [model + '_9.csv', model + '_10.csv']
        X = []
        column = pd.read_csv('dataset/DATASET_DIRECTORY/processed/train/train_1.csv', header=0, nrows=3201, delim_whitespace=False)
        X.append(column)
        for filename in filenames:
            # load data
            data_path = os.path.join(data_dir, filename)
            data = pd.read_csv(data_path, header=0, skiprows=0, nrows=3200, delim_whitespace=False)
            X.append(data)
        all_data = pd.concat(X, axis=0)
        print(all_data)
        all_data.to_csv('dataset/data_total/processed/train/train.csv', index=False, columns=None)


        return

    def generate_val_data(self, model = 'validation'):
        data_dir = "dataset/DATASET_DIRECTORY/processed/" + model
        filenames = list()
        filenames += [model + '_2.csv', model + '_3.csv', model + '_4.csv']
        filenames += [model + '_5.csv', model + '_6.csv', model + '_7.csv', model + '_8.csv']
        filenames += [model + '_9.csv', model + '_10.csv']
        X = []
        column = pd.read_csv('dataset/DATASET_DIRECTORY/processed/validation/validation_1.csv', header=0, nrows=1601, delim_whitespace=False)
        X.append(column)
        for filename in filenames:
            # load data
            data_path = os.path.join(data_dir, filename)
            data = pd.read_csv(data_path, header=0, skiprows = 0, nrows=1600, delim_whitespace=False)
            X.append(data)
        all_data = pd.concat(X, axis=0)
        print(all_data)
        all_data.to_csv('dataset/data_total/processed/validation/validation.csv', index=False, index_label = 'id')

        return


dataset_1 = autopilot_dataset()


# TrainData = dataset_1.generate_data(T=3200, seqLength=40, model = 'Train')
# train_loader = torch.utils.data.DataLoader(dataset=TrainData,batch_size=10,shuffle=False)

# ValData = dataset_1.generate_data(T=1600, seqLength=40, model = 'Validation')
# val_loader = torch.utils.data.DataLoader(dataset=ValData,batch_size=10,shuffle=False)


# TestData = dataset_1.generate_data(T=1600, seqLength=40, model ='Test')
# test_loader = torch.utils.data.DataLoader(dataset=TestData,batch_size=10,shuffle=False)



# for i, (input, output) in enumerate(train_loader):
#     print(input.shape,output.shape)

# for i, (input, output) in enumerate(val_loader):
#     print(input.shape,output.shape)

# for i, (input, output) in enumerate(test_loader):
#     print(input.shape,output.shape)

# dataset_1.generate_train_data(model = 'train')
# dataset_1.generate_val_data(model = 'validation')