import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import battery_cell_data_functions as data


def processStatic(static_data, model, typhoon_origin=False):
    """
    Script that populates the model parameters based on static test data.
    Populates OCV as a function of SOC, Q and eta.
    """

    # Searching the index of static data which corresponds with default temperature of 25 degrees celsius
    # After that, other temperature data can be calculated
    ind25 = None
    for index, single_temp_data in enumerate(static_data):
        if single_temp_data.temp == 25:
            ind25 = index
    if not ind25:
        print("There is no (default) temperature of 25 degrees celsius!")
        raise Exception()

    # Integrators that represent chgAh and disAh have a bug where they don't recognize reset signal after a certain
    # point in time during simulation, which is why I need to delete the first element for every part
    totDisAh = static_data[ind25].script1.disAh[-1] + static_data[ind25].script2.disAh[-1] + \
               static_data[ind25].script3.disAh[-1] + static_data[ind25].script4.disAh[-1] - \
               static_data[ind25].script1.disAh[0] - static_data[ind25].script2.disAh[0] - \
               static_data[ind25].script3.disAh[0] - static_data[ind25].script4.disAh[0]
    totChgAh = static_data[ind25].script1.chgAh[-1] + static_data[ind25].script2.chgAh[-1] + \
               static_data[ind25].script3.chgAh[-1] + static_data[ind25].script4.chgAh[-1] - \
               static_data[ind25].script1.chgAh[0] - static_data[ind25].script2.chgAh[0] - \
               static_data[ind25].script3.chgAh[0] - static_data[ind25].script4.chgAh[0]
    eta25 = totDisAh / totChgAh
    # Q doesn't need to be calculated differently for temp = 25

    for k in range(len(static_data)):
        # First step: calculate eta
        # scripts 1 and 3 are at test temperature but scripts 2 and 4 are always at 25 degrees
        totDisAh = static_data[k].script1.disAh[-1] + static_data[k].script2.disAh[-1] + \
                   static_data[k].script3.disAh[-1] + static_data[k].script4.disAh[-1] - \
                   static_data[k].script1.disAh[0] - static_data[k].script2.disAh[0] - \
                   static_data[k].script3.disAh[0] - static_data[k].script4.disAh[0]
        totChgAh_at_X_temp = static_data[k].script1.chgAh[-1] + static_data[k].script3.chgAh[-1] - \
                             static_data[k].script1.chgAh[0] - static_data[k].script3.chgAh[0]
        totChgAh_at_25_temp = static_data[k].script2.chgAh[-1] + static_data[k].script4.chgAh[-1] - \
                              static_data[k].script2.chgAh[0] + static_data[k].script4.chgAh[0]
        eta = (totDisAh - totChgAh_at_25_temp*eta25) / totChgAh_at_X_temp

        # Second step: calculate Q
        Q = static_data[k].script1.disAh[-1] - static_data[k].script1.disAh[0] + \
            static_data[k].script2.disAh[-1] - static_data[k].script2.disAh[0] - \
            (eta * static_data[k].script1.chgAh[-1] - eta * static_data[k].script1.chgAh[0]) - \
            (eta * static_data[k].script2.chgAh[-1] - eta * static_data[k].script2.chgAh[0])

        # Third step: calculate OCV curve
        print(f"Calculating static tests for temperature: {static_data[k].temp}")
        if typhoon_origin:  # Typhoon data
            index_discharge = list(np.where(np.array(static_data[k].script1.current) < 0)[0])
            # First voltage drop when the current starts flowing, step[1] is when the first voltage drop happens
            I_R0_dis_start = static_data[k].script1.voltage[index_discharge[0] - 1] - static_data[k].script1.voltage[index_discharge[0]]
            # Last voltage drop when the current already charged up the capacitors in the RC circuits
            I_R0_dis_end = static_data[k].script1.voltage[index_discharge[-1]] - static_data[k].script1.voltage[index_discharge[-2]]
            index_charge = list(np.where(np.array(static_data[k].script3.current) > 0)[0])
            I_R0_chg_start = static_data[k].script3.voltage[0] - static_data[k].script2.voltage[-1]
            I_R0_chg_end = static_data[k].script3.voltage[-2] - static_data[k].script3.voltage[-1]

            IR1D = min(I_R0_dis_start, I_R0_chg_start)
            IR2D = min(I_R0_dis_start, I_R0_chg_start)
            IR1C = min(I_R0_dis_start, I_R0_chg_start)
            IR2C = min(I_R0_dis_start, I_R0_chg_start)
        else:  # Colorado Boulder data scenario
            index_discharge  = np.where(np.array(static_data[k].script1.step) == 2)[0]  # index list of all slow discharge
            # First voltage drop when the current starts flowing
            I_R0_dis_start = static_data[k].script1.voltage[index_discharge[0] - 1] - static_data[k].script1.voltage[index_discharge[0]]
            # Last voltage drop when the current already charged up the capacitors in the RC circuits
            I_R0_dis_end = static_data[k].script1.voltage[index_discharge[-1] + 1] - static_data[k].script1.voltage[index_discharge[-1]]
            index_charge  = np.where(np.array(static_data[k].script3.step) == 2)[0]  # slow charge
            I_R0_chg_start = static_data[k].script3.voltage[index_charge[0]] - static_data[k].script3.voltage[index_charge[0] - 1]
            I_R0_chg_end = static_data[k].script3.voltage[index_charge[-1]] - static_data[k].script3.voltage[index_charge[-1] + 1]

            # still remains to be understood
            IR1D = min(I_R0_dis_start, 2 * I_R0_chg_end)  # For Boulder Colorado data: 0.003254
            IR2D = min(I_R0_dis_end, 2 * I_R0_chg_start)  # For Boulder Colorado data: 0.006345
            IR1C = min(I_R0_chg_start, 2 * I_R0_dis_end)  # For Boulder Colorado data: 0.012690
            IR2C = min(I_R0_chg_end, 2 * I_R0_dis_start)  # For Boulder Colorado data: 0.002928

        blend = np.array(range(len(index_discharge))) / (len(index_discharge) - 1)  # linear rise from 0 to 1 with len(discharge_index)
        IR0_blend = IR1D + (IR2D - IR1D) * blend  # IR0_blend exists because R0 measured on end and on beginning aren't necesserily the same
        # and so this list is scaled from R0 beginning to R0 end with this line
        discharged_voltage_without_IR0 = static_data[k].script1.voltage[index_discharge[0]:(index_discharge[-1] + 1)] + IR0_blend
        disZ = 1 - static_data[k].script1.disAh[index_discharge[0] : (index_discharge[-1] + 1)] / Q
        disZ = disZ + (1 - disZ[0])  # small shift in the case of measuring at the sub 100% soc

        blend = np.array(range(len(index_charge))) / (len(index_charge) - 1)
        IR0_blend = IR1C + (IR2C - IR1C) * blend
        charged_voltage_without_IR0 = static_data[k].script3.voltage[index_charge[0] : index_charge[-1] + 1] - IR0_blend
        chgZ = static_data[k].script3.chgAh[index_charge[0]:index_charge[-1] + 1] / Q * eta
        chgZ = chgZ - chgZ[0]  # Todo fix the starting value of chgZ

        Voltage_SOC_curve = interpolate.interp1d(chgZ, charged_voltage_without_IR0)
        Discharge_SOC_curve = interpolate.interp1d(disZ, discharged_voltage_without_IR0)
        deltaV50 = Voltage_SOC_curve(0.5) - Discharge_SOC_curve(0.5)

        # Now only look at values of SOC under 50%
        ind = np.where(chgZ < 0.5)[0]
        vChg = charged_voltage_without_IR0[ind] - chgZ[ind] * deltaV50
        zChg = chgZ[ind]  # Todo This part of the curve is lacking. To fix

        # Now only look at values of SOC over 50%
        ind = np.where(disZ > 0.5)[0]
        vDis = np.flipud(discharged_voltage_without_IR0[ind] + (1 - disZ[ind]) * deltaV50)
        zDis = np.flipud(disZ[ind])

        np.append(zChg, zDis)
        FULL_SOC_curve = interpolate.interp1d(np.append(zChg, zDis), np.append(vChg, vDis), fill_value="extrapolate")

        SOC_vector = np.linspace(0, 1, 201)  # output SOC_vector points for this step
        rawocv = FULL_SOC_curve(SOC_vector)

        # Final step: populate model
        model.temps.append(static_data[k].temp)
        model.etaParam_static.append(eta)
        model.QParam_static.append(Q)
        model.soc_vector.append(SOC_vector)
        model.ocv_vector.append(rawocv)

        data.plot_func([SOC_vector], [rawocv], ["OCV_SOC_static_temp_"+str(static_data[k].temp)], flag_show=False)

    plt.show()
