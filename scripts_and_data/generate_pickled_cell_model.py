import generate_battery_cell_data as cell_data
import battery_cell_functions as cell_functions
import processDynamic_script as dynamic
import processStatic_script as static
import pickle
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# This script will parse out the battery cell parameters from the battery cell data obtained from
# "generate_battery_cell_data.py" python script. The parameters will be saved in the .pickle format.

printout = True  # Print out the output model parameters
use_static_Q_eta = True  # Use parameters Q and eta from static test instead of dynamic test
data_origin = cell_data.filename
# data_origin = 'Typhoon_captured_data_hyst_0'  # 'Typhoon Hil software and hardware obtained data'
# data_origin = 'P14_Boulder_cell_data'  # 'Boulder Colorado P14 battery cell data'
output_filename = data_origin + '.pickle'  # Name of the pickled file
minimization = "double_minimize"  # "differential_evolution" / "double_minimize"
minimization = "SISOSubid"
# minimization = "triple_minimize"
# minimization = "differential_evolution"


if __name__ == "__main__":
    # Initialize model
    cell_model = cell_functions.ESC_battery_model()

    # Initialize data
    if data_origin == 'P14_Boulder_cell_data':  # test data from the Boulder university
        P14_DYN_50_P45 = cell_functions.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P45.mat"), 45)
        P14_DYN_50_P25 = cell_functions.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P25.mat"), 25)
        P14_DYN_30_P05 = cell_functions.OneTempDynData(scipy.io.loadmat("P14_DYN_30_P05.mat"), 5)
        P14_OCV_P45 = cell_functions.OneTempStaticData(scipy.io.loadmat("P14_OCV_P45.mat"), 45)
        P14_OCV_P25 = cell_functions.OneTempStaticData(scipy.io.loadmat("P14_OCV_P25.mat"), 25)
        P14_OCV_P05 = cell_functions.OneTempStaticData(scipy.io.loadmat("P14_OCV_P05.mat"), 5)

        static.processStatic([P14_OCV_P05, P14_OCV_P25, P14_OCV_P45], cell_model)
        dynamic.processDynamic([P14_DYN_30_P05, P14_DYN_50_P25, P14_DYN_50_P45], cell_model, numpoles=1, doHyst=1)

    else:  # Normal, Typhoon data format
        OCVData_full25 = scipy.io.loadmat(data_origin + ".mat")['OCVData_full25']
        OCVData_full25_voltage = OCVData_full25[2]
        DYNData_full25 = scipy.io.loadmat(data_origin + ".mat")['DYNData_full25']
        DYNData_full25_voltage = DYNData_full25[2]
        # plt.plot(OCVData_full25_voltage)
        # plt.plot(DYNData_full25_voltage)
        # plt.show()  # 2.81 is the minimum of the T25 OCV

        TYPHOON_FULL_CELL_DATA = cell_functions.CellAllData(scipy.io.loadmat(data_origin + ".mat"), [5, 25, 45], [5, 25, 45])

        static.processStatic(TYPHOON_FULL_CELL_DATA.static_data, cell_model, typhoon_origin=True)
        dynamic.processDynamic(TYPHOON_FULL_CELL_DATA.dynamic_data, cell_model, TYPHOON_FULL_CELL_DATA.numpoles, TYPHOON_FULL_CELL_DATA.doHyst, typhoon_origin=True)

    # Saving the model
    print("Saving the model")
    with open(output_filename, 'wb') as file:
        pickle.dump(cell_model, file)
        print("Model saved as ", output_filename)

    # Printing the output cell parameters, if enabled
    if printout:
        if minimization == "double_minimize" or minimization == "differential_evolution":
            print("\nMinimization algorithm used: Fminbdn for Gamma and minimize for R0, R1, RC")
        else:
            print("\nMinimization algorithm used: Fminbdn for Gamma and SISOSubid for R0, R1, RC")
        print(f"Printout of model params:\n")
        print(f"{cell_model.temps=}  Relative error: {cell_functions.error_func(cell_model.temps, 'temps')}")
        print(f"{cell_model.etaParam_static=}  Relative error: {cell_functions.error_func(cell_model.etaParam_static, 'etaParam_static')}")
        print(f"{cell_model.etaParam=}  Relative error: {cell_functions.error_func(cell_model.etaParam, 'etaParam')}")
        print(f"{cell_model.QParam_static=}  Relative error: {cell_functions.error_func(cell_model.QParam_static, 'QParam_static')}")
        print(f"{cell_model.QParam=}  Relative error: {cell_functions.error_func(cell_model.QParam, 'QParam')}")
        print(f"{cell_model.R0Param=}  Relative error: {cell_functions.error_func(cell_model.R0Param, 'R0Param')}")
        print(f"{cell_model.RParam=}  Relative error: {cell_functions.error_func(cell_model.RParam, 'RParam')}")
        print(f"{cell_model.RCParam=}  Relative error: {cell_functions.error_func(cell_model.RCParam, 'RCParam')}")
        print(f"{cell_model.M0Param=}  Relative error: {cell_functions.error_func(cell_model.M0Param, 'M0Param')}")
        print(f"{cell_model.MParam=}  Relative error: {cell_functions.error_func(cell_model.MParam, 'MParam')}")
        print(f"{cell_model.GParam=}  Relative error: {cell_functions.error_func(cell_model.GParam, 'GParam')}")
        print(f"cell_model.ocv_vector at 25 degrees RMS error: {cell_functions.error_func(cell_model.ocv_vector[1], 'OCV')}")
        try:
            cell_functions.plot_func([cell_model.soc_vector[1], cell_data.SOC_default],
                                     [cell_model.ocv_vector[1], cell_data.OCV_default[1]],
                                     [f"OCV vs SOC graph (Colorado, octave vs {data_origin}) for 25 celsius",
                                      f"OCV vs SOC graph (Colorado, octave vs {data_origin}) for 25 celsius"],
                                     flag_show=False)
            cell_functions.plot_func([cell_data.SOC_default],
                                     [np.array(cell_data.OCV_default[0]) - np.array(cell_model.ocv_vector[0])],
                                     ['T5 RMS error in OCV [V] as a function of SOC'],
                                     flag_show=False)
            cell_functions.plot_func([cell_data.SOC_default],
                                     [np.array(cell_data.OCV_default[1]) - np.array(cell_model.ocv_vector[1])],
                                     ['T25 RMS error in OCV [V] as a function of SOC'],
                                     flag_show=False)
            cell_functions.plot_func([cell_data.SOC_default],
                                     [np.array(cell_data.OCV_default[2]) - np.array(cell_model.ocv_vector[2])],
                                     ['T45 RMS error in OCV [V] as a function of SOC'],
                                     flag_show=False)
        except:
            print(f"Unable to plot {data_origin}")
            pass
