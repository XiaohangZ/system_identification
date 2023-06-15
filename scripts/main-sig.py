import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import sys
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


o_path = os.getcwd()
sys.path.append(o_path)

from metrics import * 
from models.nn import *
from config import device
from utils import * 
from train import *
from datasets.autopilot.AutopilotData import *
from datasets.patrolship.DataLoader_all import *

def main() -> None:

    TrainData = autopilot_dataset().get_data(T=3200, seqLength=10)
    train_loader = torch.utils.data.DataLoader(dataset=TrainData,batch_size=16,shuffle=False)

    ValData = autopilot_dataset().get_data(T=1600, seqLength=10)
    val_loader = torch.utils.data.DataLoader(dataset=ValData,batch_size=16,shuffle=False)
    print(len(val_loader))

    TestData = autopilot_dataset().get_data(T=1600, seqLength=10)
    test_loader = torch.utils.data.DataLoader(dataset=TestData,batch_size=16,shuffle=False)

    learning_rate = .001
    num_epochs = 50
    optimizer = torch.optim.Adam(FC().parameters(), lr=learning_rate)

    model = FC()

    train(model = model, SavingName='./checkpoints/nn.ckpt', train_loader = train_loader, val_loader=val_loader,num_epochs = num_epochs)
    test(model = model, SavingName='./checkpoints/nn.ckpt', test_loader=test_loader)

main()




