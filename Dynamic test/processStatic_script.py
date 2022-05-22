import numpy as np
from scipy import interpolate


def processStatic(OCVData, model):
    """

    """
    SOC = np.linspace(0, 1, 201)  # output SOC points for this step
    for k in range(len(OCVData)):
        # Todo: make Q and eta calculation work well for temps other than default
        # First step: calculate eta
        totDisAh = OCVData[k].script1.disAh[-1] + OCVData[k].script2.disAh[-1] + OCVData[k].script3.disAh[-1] + OCVData[k].script4.disAh[-1]
        totChgAh = OCVData[k].script1.chgAh[-1] + OCVData[k].script2.chgAh[-1] + OCVData[k].script3.chgAh[-1] + OCVData[k].script4.chgAh[-1]
        eta = totDisAh / totChgAh
        model.temps.append(eta)

        # Second step: calculate Q
        Q = OCVData[k].script1.disAh[-1] + OCVData[k].script2.disAh[-1] - eta * OCVData[k].script1.chgAh[-1] - eta * OCVData[k].script2.chgAh[-1]
        model.temps.append(Q)

        # Third step: calculate OCV curve
        print(f"Calculating static tests for temperature: {OCVData[k].temp}")
        indD  = np.where(np.array(OCVData[k].script1.step) == 2)[0]  # slow discharge
        IR1Da = OCVData[k].script1.voltage[indD[0]-1] - OCVData[k].script1.voltage[indD[0]]
        IR2Da = OCVData[k].script1.voltage[indD[-1]+1] - OCVData[k].script1.voltage[indD[-1]]

        indC  = np.where(np.array(OCVData[k].script3.step) == 2)[0]  # slow charge
        IR1Ca = OCVData[k].script3.voltage[indC[0]] - OCVData[k].script3.voltage[indC[0]-1]
        IR2Ca = OCVData[k].script3.voltage[indC[-1]] - OCVData[k].script3.voltage[indC[-1]+1]
        IR1D = min(IR1Da, 2*IR2Ca)
        IR2D = min(IR2Da, 2*IR1Ca)
        IR1C = min(IR1Ca, 2*IR2Da)
        IR2C = min(IR2Ca, 2*IR1Da)

        blend = np.array(range(len(indD)-1))/(len(indD)-1)
        IRblend = IR1D + (IR2D-IR1D)*blend
        disV = OCVData[k].script1.voltage[indD[0]:indD[-1]] + IRblend
        disZ = 1 - OCVData[k].script1.disAh[indD[0]:indD[-1]]/Q
        disZ = disZ + (1 - disZ[0])
        # Todo: checkpoint, everything works, just disZ[-1] is not exactly the same as in octave

        blend = np.array(range(len(indC)-1))/(len(indC)-1)
        IRblend = IR1C + (IR2C-IR1C)*blend
        chgV = OCVData[k].script3.voltage[indC[0]:indC[-1]] - IRblend
        chgZ = OCVData[k].script3.chgAh[indC[0]:indC[-1]]/Q
        chgZ = chgZ - chgZ[0]

        Voltage_SOC_curve = interpolate.interp1d(chgZ, chgV)
        Discharge_SOC_curve = interpolate.interp1d(disZ, disV)
        deltaV50 = Voltage_SOC_curve(0.5) - Discharge_SOC_curve(0.5)
        ind = np.where(chgZ < 0.5)[0]
        vChg = chgV[ind[0]:ind[-1]] - chgZ[ind[0]:ind[-1]]*deltaV50
        zChg = chgZ[ind[0]:ind[-1]]
        ind = np.where(disZ > 0.5)[0]
        vDis = np.flipud(disV[ind[0]:ind[-1]] + (1 - disZ[ind[0]:ind[-1]])*deltaV50)
        zDis = np.flipud(disZ[ind[0]:ind[-1]])
        np.append(zChg, zDis)
        FULL_SOC_curve = interpolate.interp1d(np.append(zChg, zDis), np.append(vChg, vDis))
        rawocv = FULL_SOC_curve(SOC, 'extrapolate')
        model.soc_vector.append(SOC)
        model.ocv_vector.append(rawocv)