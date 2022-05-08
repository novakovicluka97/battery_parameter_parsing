from battery_cell_data import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
fminbnd = scipy.optimize.fminbound
import processDynamic


# setupDynData call this function
numpoles = 1  # number of resistor--capacitor pairs in final model
doHyst = 1    # whether to include hysteresis in model

proccessDynamic([P14_DYN_04_N25, P14_DYN_05_N15], P14_model, numpoles, doHyst)
print(1)