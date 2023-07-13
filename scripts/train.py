import os
import sys
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt 

o_path = os.getcwd()
sys.path.append(o_path)

from metrics import *
from config import device
from utils import *

learning_rate = .001
num_epochs = 125

def train(model = None,SavingName=None, train_loader = None, val_loader=None, num_epochs = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)

    accuracy_mean = []
    accuracy_change=[]
    epoch_change=[]

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
            
            
            if epoch%25 == 0:
                with torch.no_grad():
                    model.eval()     
                    
                    pred,gt = [],[]
                    
                    for signalsV, labelsV in val_loader:
                        
                        labelsV = labelsV.to(device)
                        signalsV = signalsV.to(device)
                        
                        outputsV = model(signalsV)
                        
                        gt.extend(labelsV.cpu().numpy()[0])
                        pred.extend(outputsV[0].round().cpu().numpy())
                    
                    gt = np.asarray(gt,np.float32)
                    pred = np.asarray(pred)
                    # print('MSE of the model on the {} epoch: {} %'.format(epoch,accuracy(pred,gt)))

                    accuracy_mean.append(accuracy(pred,gt))
                    accuracy_25 = accuracy_mean[epoch:]                   
                    # print(len(accuracy_10))

                    
                    if len(accuracy_25) == 25:
                      a = 0
                      for i in accuracy_25:
                        a = i + a
                      print('Mean MSE of the model on the {} epoch: {} %'.format(epoch,a/len(train_loader)))
                      epoch_change.append(epoch)
                      accuracy_change.append(a/len(train_loader))
                      print(epoch_change,accuracy_change)

                      if len(accuracy_change)==(num_epochs/25):
                        # learning process visualization
                        plt.title('Training Accuracy')
                        plt.ylabel('Accuracy')
                        plt.xlabel('Epoch')
                        plt.grid(True)
                        plt.autoscale(axis='x', tight=True)
                        plt.plot(epoch_change,accuracy_change)
                        plt.show()

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

        print('MSE of the model test samples: {} %'.format(accuracy(pred,gt)))