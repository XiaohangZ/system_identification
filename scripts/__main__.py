import sys
import os

o_path = os.getcwd()
sys.path.append(o_path)

from train import *
from dataset.data_load.autopilot.data_loader import *
from deepsysid.models_basic.nn import *

def main() -> None:

    learning_rate = .001
    num_epochs = 125
    optimizer = torch.optim.Adam(FC().parameters(), lr=learning_rate)

    model = FC()

    train(model = model, SavingName='./checkpoints/nn.ckpt', train_loader = train_loader, val_loader=val_loader,num_epochs = num_epochs)
    test(model = model, SavingName='./checkpoints/nn.ckpt', test_loader=test_loader)

main()




