import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Dynamic and static data created this model
# E2 Li-ion model

E2model = scipy.io.loadmat("E2model.mat")
OCV0 = E2model['model']['OCV0'][0][0][0]
OCVrel = E2model['model']['OCVrel'][0][0][0]
SOC = E2model['model']['SOC'][0][0][0]
temps = E2model['model']['temps'][0][0][0]
OCV = []
for index, temp in enumerate(temps):
    OCV.append(list(np.array(OCV0) + temp*np.array(OCVrel)))

QParam = E2model['model']['QParam'][0][0][0]
etaParam = E2model['model']['etaParam'][0][0][0]
RCParam = E2model['model']['RCParam'][0][0]
RCParam_formated = []
for i in RCParam:
    RCParam_formated.append(i[0])
RCParam = RCParam_formated
R0Param = E2model['model']['R0Param'][0][0][0]
MParam = E2model['model']['MParam'][0][0][0]
M0Param = E2model['model']['M0Param'][0][0][0]
GParam = E2model['model']['GParam'][0][0][0]

# data to compare our model with
time = []
current = []
initial_SOC = 100  # %
initial_V_RC = 0
initial_V_hysteresis = 0

vest = scipy.io.loadmat("vest.mat")['vest']
time_vest = scipy.io.loadmat("time_vest.mat")['time_vest']
expected_voltage_25 = []
time_voltage = []
for index, i in enumerate(vest):
    expected_voltage_25.append(i[0])
    time_voltage.append(time_vest[0][index])

if __name__ == "__main__":
    print(R0Param, RCParam, QParam, temps)
    plt.figure(figsize=(11, 11))

    plt.subplot(221)
    plt.plot(temps, QParam)
    plt.plot(temps, RCParam)
    plt.plot(temps, R0Param)
    plt.plot(temps, etaParam)
    plt.subplot(222)
    plt.plot(temps, GParam)
    plt.plot(temps, M0Param)
    plt.plot(temps, MParam)
    plt.subplot(223)
    plt.plot(SOC, OCV0)
    plt.plot(SOC, OCV[3])
    plt.plot(SOC, OCV[7])
    plt.subplot(224)
    plt.plot(time_voltage, expected_voltage_25)
    plt.show()

