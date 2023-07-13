import os
import sys

o_path = os.getcwd()
sys.path.append(o_path)

import matplotlib.pyplot as plt

from dataset.data_load.autopilot.data_loader import *
from deepsysid.models.lstm import *
from utils import * 
from metrics import * 
from config import device


def train(model = None, SavingName=None, train_loader = None, val_loader=None, optimizer = None):
       
    total_step = len(train_loader)
    accuracy_mean = []
    accuracy_change=[]
    epoch_change=[]
    for epoch in range(num_epochs):
        for i, (signals, labels) in enumerate(train_loader):

            # Forward pass
            outputs = model(signals)
            
            loss_function = nn.MSELoss()
            #loss_function = nn.CrossEntropyLoss()
            loss = loss_function(labels.to(torch.float64), outputs.to(torch.float64))

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
                    
                    gt = np.asarray(gt,np.float64)
                    pred = np.asarray(pred)
                    #print('Val Accuracy of the model on the {} epoch: {} %'.format(epoch,accuracy(pred,gt)))

                    accuracy_mean.append(accuracy(pred,gt))
                    accuracy_25 = accuracy_mean[epoch:]                   
                    #print(accuracy_10)

                    
                    if len(accuracy_25) == 25:
                      a = 0
                      for i in accuracy_25:
                        a = i + a
                      print('Mean Val Accuracy of the model on the {} epoch: {} %'.format(epoch,a/len(train_loader)))
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

        print('Test Accuracy of the model test samples: {} %'.format(accuracy(pred,gt)))


    
RCS= FCLSTM()
#RCS= FC()

learning_rate = .001
num_epochs = 125
optimizer = torch.optim.Adam(RCS.parameters(), lr=learning_rate)

train(model = RCS, SavingName='./checkpoints/nn.ckpt', train_loader = train_loader, val_loader=val_loader, optimizer = optimizer)
test(model = RCS, SavingName='./checkpoints/nn.ckpt', test_loader=test_loader)