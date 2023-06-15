import torch
import torch.nn as nn
import numpy as np
import os
import sys

o_path = os.getcwd()
sys.path.append(o_path)

from metrics import * 
from models.nn import *
from config import *
from utils import * 
from datasets.autopilot.AutopilotData import *
from datasets.patrolship.DataLoader_all import *

learning_rate = .001
num_epochs = 50


def train(model = None,SavingName=None, train_loader = None, val_loader=None, num_epochs = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (input, output) in enumerate(train_loader):

            # Forward pass
            outputs = model(input)
            
            loss_function = nn.MSELoss()
            loss = loss_function(output.to(torch.float32), outputs.to(torch.float32))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            
            
            if epoch%10 == 0:
                with torch.no_grad():
                    model.eval()     
                    
                    pred,gt = [],[]
                    
                    for signalsV, labelsV in val_loader:
                        
                        labelsV = labelsV.to(device)
                        signalsV = signalsV.to(device)
                        
                        outputsV = model(signalsV)
                        
                        gt.extend(labelsV.gpu().numpy()[0])
                        pred.extend(outputsV[0].round().gpu().numpy())
                    
                    gt = np.asarray(gt,np.float32)
                    pred = np.asarray(pred)
                        
                    print('Val Accuracy of the model on the {} epoch: {} %'.format(epoch,accuracy(pred,gt)))
                    
                model.train()
            
    # Save the model checkpoint
    checkDirMake(os.path.dirname(SavingName))
    torch.save(model.state_dict(), SavingName)
    

def test(model = None,SavingName=None, test_loader=None):
    model.load_state_dict(torch.load(SavingName))
    # Test the model
    
    model.eval() 
    with torch.no_grad():
        
        pred,gt = [],[]
        
        for signals, labels in test_loader:
            
            signals = signals.to(device)
            outputs = model(signals)
            outputs = outputs.round().cpu().numpy()
             
            gt.extend(labels.cpu().numpy()[0])
            pred.extend(outputs[0])
        
        gt = np.asarray(gt,np.float32)
        pred = np.asarray(pred)

        print('Test Accuracy of the model test samples: {} %'.format(accuracy(pred,gt)))

def main() -> None:

    TrainData = autopilot_dataset().get_data(T=320, seqLength=10)
    train_loader = torch.utils.data.DataLoader(dataset=TrainData,batch_size=16,shuffle=False)

    ValData = autopilot_dataset().get_data(T=160, seqLength=10)
    val_loader = torch.utils.data.DataLoader(dataset=ValData,batch_size=16,shuffle=False)
    print(len(val_loader))

    TestData = autopilot_dataset().get_data(T=160, seqLength=10)
    test_loader = torch.utils.data.DataLoader(dataset=TestData,batch_size=16,shuffle=False)

    learning_rate = .001
    num_epochs = 50
    optimizer = torch.optim.Adam(FC().parameters(), lr=learning_rate)

    model = FC()

    train(model = model, SavingName='./checkpoints/nn.ckpt', train_loader = train_loader, val_loader=val_loader,num_epochs = num_epochs)
    test(model = model, SavingName='./checkpoints/nn.ckpt', test_loader=test_loader)

main()