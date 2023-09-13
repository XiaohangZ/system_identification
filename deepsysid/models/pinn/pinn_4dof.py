import torch.nn as nn
import torch
from torch.utils import data
from typing import Dict, List, Literal, Optional, Tuple
from numpy.typing import NDArray
import numpy as np


class PINNNet(nn.Module):
    def __init__(self, inputNode=7, hiddenNode=256, outputNode=4):
        super(PINNNet, self).__init__()
        # Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode
        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        self.activation = torch.nn.Sigmoid()

    def forward(self, X):
        out1 = self.Linear1(X)
        out2 = self.activation(out1)
        out3 = self.Linear2(out2)
        return out3

def pinn_loss_4dof(u, v, p, r, phi, u_prev, v_prev, p_prev, r_prev):
    # Constant
    rho_water = 1025.0
    g = 9.81

    # Main Particulars
    Lpp = 51.5
    B = 8.6
    D = 2.3

    #Load condition
    disp = 355.88
    m = 365.79 * 10 ^ 3
    Izz = 3.3818 * 10 ^ 7
    Ixx = 3.4263 * 10 ^ 6
    gm = 1.0
    LCG = 20.41
    VCG = 3.36
    xG = -3.38
    zG = -1.06

    # Data for surge equation
    Xudot = -17400.0
    Xuau = -1960.0
    Xvr = 0.33 * m

    # Hydrodynamic coefficients in sway equation
    Yvdot = -1.9022 * 10 ^ 6
    Ypdot = -0.296 * 10 ^ 6
    Yrdot = -1.4 * 10 ^ 6
    Yauv = -11800
    Yur = 131000
    Yvav = -3700
    Yrar = 0
    Yvar = -794000
    Yrav = -182000
    Ybauv = 10800
    Ybaur = 251000
    Ybuu = -74

    #Hydrodynamic coefficients in roll equation
    Kvdot = 296000
    Kpdot = -674000
    Krdot = 0
    Kauv = 9260
    Kur = -102000
    Kvav = 29300
    Krar = 0
    Kvar = 621000
    Krav = 142000
    Kbauv = -8400
    Kbaur = -196000
    Kbuu = -1180
    Kaup = -15500
    Kpap = -416000
    Kp = -500000
    Kb = 0
    Kbbb = -0.325 * rho_water * g * disp

    # Hydrodynamic coefficients in yaw equation
    Nvdot = 538000
    Npdot = 0
    Nrdot = -4.3928 * 10 ^ 7
    Nauv = -92000
    Naur = -4710000
    Nvav = 0
    Nrar = -202000000
    Nvar = 0
    Nrav = -15600000
    Nbauv = -214000
    Nbuar = -4980000
    Nbuau = -8000

    #Auxiliary variables
    b = phi
    au = abs(u)
    av = abs(v)
    ar = abs(r)
    ap = abs(p)

    #Total Mass Matrix
    M = np.array([(m-Xudot),0,0,0,0,0],
                 [0,(m-Yvdot),-(m*zG+Ypdot),(m*xG-Yrdot),0,0]
                 [0,-(m*zG+Kvdot),(Ixx-Kpdot),-Krdot,0,0]
                 [0,(m*xG-Nvdot),-Npdot,(Izz-Nrdot),0,0]
                 [0,0,0,0,1,0]
                 [0,0,0,0,0,1])

    # Hydrodynamic forces without added mass terms (considered in the M matrix)
    Xh = Xuau * u * au + Xvr * v * r

    Yh = (Yauv * au * v + Yur * u * r + Yvav * v * av + Yvar * v * ar + Yrav * r * av
          + Ybauv * b * abs(u * v) + Ybaur * b * abs(u * r) + Ybuu * b * u ^ 2)

    Kh = Kauv * au * v + Kur * u * r + Kvav * v * av + Kvar * v * ar + Krav * r * av
    + Kbauv * b * abs(u * v) + Kbaur * b * abs(u * r) + Kbuu * b * u ^ 2 + Kaup * au * p
    + Kpap * p * ap + Kp * p + Kbbb * b ^ 3 - (rho_water * g * gm * disp) * b
    + Kb * b

    Nh = Nauv * au * v + Naur * au * r + Nrar * r * ar + Nrav * r * av
    +Nbauv * b * abs(u * v) + Nbuar * b * u * ar + Nbuau * b * u * au

    #Rigid - body centripetal accelerations
    Xc = m * (r * v + xG * r ^ 2 - zG * p * r)
    Yc = - m * u * r
    Kc = m * zG * u * r
    Nc = - m * xG * u * r

    #Total forces
    Xe = 0
    Ye = 0
    Ke = 0
    Ne = 0
    F1 = Xh + Xc + Xe
    F2 = Yh + Yc + Ye
    F4 = Kh + Kc + Ke
    F6 = Nh + Nc + Ne

    force = np.array([F1,F2,F4,F6])
    M_1 = np.linalg.inv(M)
    F = np.array([F1,F2,F4,F6,p,r])
    sampling_time = 1
    acceleration_pred = np.dot(F, M_1)
    u_dot = acceleration_pred[0]
    v_dot = (acceleration_pred[1]^2 + acceleration_pred[2]^2 + acceleration_pred[3]^2)**0.5
    p_dot = acceleration_pred[4]
    r_dot = acceleration_pred[5]
    acceleration_pred = np.array([u_dot, v_dot, p_dot, r_dot])
    acceleration_true = np.array([(u-u_prev)/sampling_time, (v-v_prev)/sampling_time, (p-p_prev)/sampling_time, (r-r_prev)/sampling_time])
    loss = acceleration_true - acceleration_pred
    MSE_R = 0
    for l in loss:
        MSE_R = MSE_R + l^2
    return MSE_R

class RecurrentPINNDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
            self,
            control_seqs: List[NDArray[np.float64]],
            state_seqs: List[NDArray[np.float64]],
            sequence_length: int,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
            self,
            control_seqs: List[NDArray[np.float64]],
            state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - self.sequence_length - 1) / self.sequence_length
            )

            x = np.zeros(
                (n_samples, self.sequence_length, self.control_dim),
                dtype=np.float64,
            )
            y = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x[idx, :, :] = control[time: time + self.sequence_length, :]
                y[idx, :, :] = state[time: time + self.sequence_length, :]

            x_seq.append(x)
            y_seq.append(y)

        return np.vstack(x_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {'x': self.x[idx], 'y': self.y[idx]}

