from fractions import Fraction
import math
import random
import torch
from _operator import truediv
import torch.utils.data as Data


class autopilot_dataset:
    #input time duration
    def __init__(self) -> None:
        pass

    def get_data(self, T, seqLength = None):
        #define initial state
        U = random.uniform(5, 12)

        delta_max = 30*math.pi/180
        dot_delta_max = 10*math.pi/180
        delta = random.uniform(-delta_max,delta_max)
        #print(U,delta)

        U_list=[]
        delta_list=[]
        r_list=[]
        t_list=[]
        t=0

        while t < T:
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

            r = 0 
            F_rudder = K_delta*delta + K_U_delta*U*delta + K_U2_delta*math.pow(U,2)*delta + K_U3_delta*math.pow(U,3)*delta
            F_hydro = N_r*r + N_r3*math.pow(r,3) + N_U_r3*U*math.pow(r,3) + N_U2_r3*math.pow(U,2)*math.pow(r,3) + N_U3_r3*math.pow(U,3)*math.pow(r,3)
            dot_r = (F_rudder - F_hydro) / (T_dotr + T_U_dotr*U + T_U2_dotr*math.pow(U,2) + T_U3_dotr*math.pow(U,3))
            r = r + 0.01*dot_r
            
            t = t+1
            U_list.append(U)
            delta_list.append(delta)
            r_list.append(r)
            t_list.append(t)

            dot_delta = random.uniform(-dot_delta_max, dot_delta_max) 
            delta = delta + dot_delta

            U_tensor = torch.tensor(U_list)
            delta_tensor = torch.tensor(delta_list)
            r_tensor = torch.tensor(r_list)
            data_tensor = torch.stack([U_tensor, delta_tensor, r_tensor], dim=1)
            input_tensor = torch.stack([U_tensor, delta_tensor], dim=1)
            input_list = input_tensor.tolist()
            output_tensor = r_tensor
            output_list = r_list


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

          
        return torch_dataset
    
#print(autopilot_dataset().get_data(T=320, seqLength=10))

TrainData = autopilot_dataset().get_data(T=320, seqLength=10)
train_loader = torch.utils.data.DataLoader(dataset=TrainData,batch_size=16,shuffle=False)

print(len(train_loader))

for i, (input, output) in enumerate(train_loader):
    print(input.shape,output.shape)

