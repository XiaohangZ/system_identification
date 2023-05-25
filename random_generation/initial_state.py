import random
import math
#from fractions import Fraction


class generate_initial_state:
    def __init__(self) -> None:
        pass

    def initial_parameter (self):
        U_c = random.uniform(0, 100)

        delta_max = 30*math.pi/180
        #dotdelta_max = 10*math.pi/180
        delta = random.uniform(-delta_max,delta_max)
        #print(U_c,delta)

        return U_c, delta
    
#print(generate_initial_state().initial_parameter())
 
     