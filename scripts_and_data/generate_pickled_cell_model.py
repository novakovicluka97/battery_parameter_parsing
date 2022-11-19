import generate_battery_cell_data as model_data
import battery_cell_data_functions as data
import processDynamic_script as dynamic
import processStatic_script as static
import pickle
import scipy.io
import matplotlib.pyplot as plt


# This script will parse out the battery cell parameters from the battery cell data obtained from
# "generate_battery_cell_data.py" python script. The parameters will be saved in the .pickle format.

printout = True  # Print out the output model parameters
use_static_Q_eta = True  # Use parameters Q and eta from static test instead of dynamic test
data_origin = 'Typhoon_captured_data'  # 'Typhoon Hil software and hardware obtained data'
data_origin = 'P14_Boulder_cell_data'  # 'Boulder Colorado P14 battery cell data'
output_filename = data_origin + '.pickle'  # Name of the pickled file


if __name__ == "__main__":
    # Initialize model
    cell_model = data.ESC_battery_model()

    # Initialize data
    if data_origin == 'P14_Boulder_cell_data':  # test data from the Boulder university
        P14_DYN_50_P45 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P45.mat"), 45)
        P14_DYN_50_P25 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P25.mat"), 25)
        P14_DYN_30_P05 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_30_P05.mat"), 5)
        P14_OCV_P45 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P45.mat"), 45)
        P14_OCV_P25 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P25.mat"), 25)
        P14_OCV_P05 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P05.mat"), 5)

        static.processStatic([P14_OCV_P05, P14_OCV_P25, P14_OCV_P45], cell_model)
        dynamic.processDynamic([P14_DYN_30_P05, P14_DYN_50_P25, P14_DYN_50_P45], cell_model, numpoles=1, doHyst=1)

    else:  # Normal, Typhoon data format
        TYPHOON_FULL_CELL_DATA = data.CellAllData(scipy.io.loadmat(data_origin + ".mat"), [5, 25, 45], [5, 25, 45])

        static.processStatic(TYPHOON_FULL_CELL_DATA.static_data, cell_model, typhoon_origin=True)
        dynamic.processDynamic(TYPHOON_FULL_CELL_DATA.dynamic_data, cell_model, TYPHOON_FULL_CELL_DATA.numpoles, TYPHOON_FULL_CELL_DATA.doHyst, typhoon_origin=True)

    # Saving the model
    print("Saving the model")
    with open(output_filename, 'wb') as file:
        pickle.dump(cell_model, file)
        print("Model saved as ", output_filename)

    # Printing the output cell parameters, if enabled
    if printout:
        print(f"\nPrintout of model params:\n")
        print(f"{cell_model.temps=}  Relative error: {model_data.error_func(cell_model.temps, 'temps')}")
        print(f"{cell_model.etaParam=}  Relative error: {model_data.error_func(cell_model.etaParam, 'etaParam')}")
        print(f"{cell_model.R0Param=}  Relative error: {model_data.error_func(cell_model.R0Param, 'R0Param')}")
        print(f"{cell_model.QParam=}  Relative error: {model_data.error_func(cell_model.QParam, 'QParam')}")
        print(f"{cell_model.RParam=}  Relative error: {model_data.error_func(cell_model.RParam, 'RParam')}")
        print(f"{cell_model.RCParam=}  Relative error: {model_data.error_func(cell_model.RCParam, 'RCParam')}")
        print(f"{cell_model.etaParam_static=}  Relative error: {model_data.error_func(cell_model.etaParam_static, 'etaParam_static')}")
        print(f"{cell_model.QParam_static=}  Relative error: {model_data.error_func(cell_model.QParam_static, 'QParam_static')}")
        print(f"{cell_model.M0Param=}  Relative error: {model_data.error_func(cell_model.M0Param, 'M0Param')}")
        print(f"{cell_model.MParam=}  Relative error: {model_data.error_func(cell_model.MParam, 'MParam')}")
        print(f"{cell_model.GParam=}  Relative error: {model_data.error_func(cell_model.GParam, 'GParam')}")
        plt.plot(cell_model.soc_vector[1], cell_model.ocv_vector[1])
        plt.plot(model_data.SOC_default, model_data.OCV_default[1])  # OCV curve
        plt.title(f"OCV vs SOC graph (Colorado, octave vs {data_origin}) for 25 celsius")
        plt.show()
