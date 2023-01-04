import generate_battery_cell_data as cell_data
import pickle
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# This script will parse out the battery cell parameters from the battery cell data obtained from
# "generate_battery_cell_data.py" python script. The parameters will be saved in the .pickle format.

use_static_Q_eta = True                            # Use parameters Q and eta from static test instead of dynamic test
data_origin = cell_data.filename
# data_origin = 'Typhoon_captured_data_hyst_0'     # 'Typhoon Hil software and hardware obtained data'
# data_origin = 'P14_Boulder_cell_data'            # 'Boulder Colorado P14 battery cell data'
output_filename = data_origin + '.pickle'          # Name of the pickled file
minimization = "SISOSubid"
# minimization = "double_minimize"
# minimization = "triple_minimize"
# minimization = "differential_evolution"
init_guess = "correct"                             # "correct" replaces all output dynamic parameters into ones provided
init_guess = "random"
numpoles = 1

error_calc_range = "full"                          # full optimizes for the error over the entire range of values
# error_calc_range = "other"                         # alternative calculates between 5% and 95% SOC


if __name__ == "__main__":
    # to avoid circular imports we import some functions here
    import battery_cell_functions as cell_functions
    import processDynamic_script as dynamic
    import processStatic_script as static

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
        # dynamic.processDynamic(TYPHOON_FULL_CELL_DATA.dynamic_data, cell_model, TYPHOON_FULL_CELL_DATA.numpoles, TYPHOON_FULL_CELL_DATA.doHyst, typhoon_origin=True)
        dynamic.processDynamic(TYPHOON_FULL_CELL_DATA.dynamic_data, cell_model, numpoles, TYPHOON_FULL_CELL_DATA.doHyst, typhoon_origin=True)

        cell_functions.save_and_show_data(model=cell_model, dynamic_data=TYPHOON_FULL_CELL_DATA.dynamic_data, numpoles=numpoles)

    # Saving the model
    print("Saving the model")
    with open(output_filename, 'wb') as file:
        pickle.dump(cell_model, file)
        print("Model saved as ", output_filename)

