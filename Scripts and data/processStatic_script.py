import numpy as np
from scipy import interpolate


def processStatic(static_data, model):
    """
    Script that populates the model parameters based on static test data
    """
    SOC_vector = np.linspace(0, 1, 201)  # output SOC_vector points for this step

    # We need to find the index of static data which coresponds with default temperature of 25 degrees celsius
    # After that, other temperatures can be calculated
    ind25 = None
    for index, single_temp_data in enumerate(static_data):
        if single_temp_data.temp == 25:
            ind25 = index
    if not ind25:
        print("There is no default temperature of 25 degrees celsius!")
        raise Exception()

    totDisAh = static_data[ind25].script1.disAh[-1] + static_data[ind25].script2.disAh[-1] + \
               static_data[ind25].script3.disAh[-1] + static_data[ind25].script4.disAh[-1]
    totChgAh = static_data[ind25].script1.chgAh[-1] + static_data[ind25].script2.chgAh[-1] + \
               static_data[ind25].script3.chgAh[-1] + static_data[ind25].script4.chgAh[-1]
    eta25 = totDisAh / totChgAh
    # Q doesn't need to be calculated differently for temp = 25

    for k in range(len(static_data)):
        # First step: calculate eta
        # scripts 1 and 3 are at test temperature but scripts 2 and 4 are always at 25 degrees
        totDisAh = static_data[k].script1.disAh[-1] + static_data[k].script2.disAh[-1] + static_data[k].script3.disAh[-1] + static_data[k].script4.disAh[-1]
        totChgAh_at_X_temp = static_data[k].script1.chgAh[-1] + static_data[k].script3.chgAh[-1]
        totChgAh_at_25_temp = static_data[k].script2.chgAh[-1] + static_data[k].script4.chgAh[-1]
        eta = (totDisAh - totChgAh_at_25_temp*eta25) / totChgAh_at_X_temp

        # Second step: calculate Q
        Q = static_data[k].script1.disAh[-1] + static_data[k].script2.disAh[-1] - eta * static_data[k].script1.chgAh[-1] - eta * static_data[k].script2.chgAh[-1]

        # Third step: calculate OCV curve
        print(f"Calculating static tests for temperature: {static_data[k].temp}")
        indD  = np.where(np.array(static_data[k].script1.step) == 2)[0]  # slow discharge
        IR1Da = static_data[k].script1.voltage[indD[0] - 1] - static_data[k].script1.voltage[indD[0]]
        IR2Da = static_data[k].script1.voltage[indD[-1] + 1] - static_data[k].script1.voltage[indD[-1]]

        indC  = np.where(np.array(static_data[k].script3.step) == 2)[0]  # slow charge
        IR1Ca = static_data[k].script3.voltage[indC[0]] - static_data[k].script3.voltage[indC[0] - 1]
        IR2Ca = static_data[k].script3.voltage[indC[-1]] - static_data[k].script3.voltage[indC[-1] + 1]
        IR1D = min(IR1Da, 2*IR2Ca)
        IR2D = min(IR2Da, 2*IR1Ca)
        IR1C = min(IR1Ca, 2*IR2Da)
        IR2C = min(IR2Ca, 2*IR1Da)

        blend = np.array(range(len(indD)))/(len(indD)-1)
        IRblend = IR1D + (IR2D-IR1D)*blend
        disV = static_data[k].script1.voltage[indD[0]:(indD[-1] + 1)] + IRblend
        disZ = 1 - static_data[k].script1.disAh[indD[0]:(indD[-1] + 1)] / Q
        disZ = disZ + (1 - disZ[0])

        blend = np.array(range(len(indC)))/(len(indC)-1)
        IRblend = IR1C + (IR2C-IR1C)*blend
        chgV = static_data[k].script3.voltage[indC[0]:indC[-1] + 1] - IRblend
        chgZ = static_data[k].script3.chgAh[indC[0]:indC[-1] + 1] / Q
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
        FULL_SOC_curve = interpolate.interp1d(np.append(zChg, zDis), np.append(vChg, vDis), fill_value="extrapolate")
        rawocv = FULL_SOC_curve(SOC_vector)

        # Final step: populate model
        model.temps.append(static_data[k].temp)
        model.etaParam_static.append(eta)
        model.QParam_static.append(Q)
        model.soc_vector.append(SOC_vector)
        model.ocv_vector.append(rawocv)
