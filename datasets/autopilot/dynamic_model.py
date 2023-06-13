from fractions import Fraction
import math

class dynamic_model:
    def __init__(self) -> None:
        pass

    def get_R(self, U, delta, r, dotr):
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

        F_rudder_hydro_nn = (T_dotr + T_U_dotr*U + T_U2_dotr*math.pow(U,2) + T_U3_dotr*math.pow(U,3))*dotr
        F_rudder = K_delta*delta + K_U_delta*U*delta + K_U2_delta*math.pow(U,2)*delta + K_U3_delta*math.pow(U,3)*delta
        F_hydro = N_r*r + N_r3*math.pow(r,3) + N_U_r3*U*math.pow(r,3) + N_U2_r3*math.pow(U,2)*math.pow(r,3) + N_U3_r3*math.pow(U,3)*math.pow(r,3)

        R = F_rudder - F_hydro -F_rudder_hydro_nn
        # print(R)

        return R
    


