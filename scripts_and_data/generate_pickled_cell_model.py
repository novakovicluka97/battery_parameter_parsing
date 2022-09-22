import battery_cell_data_functions as data
import processDynamic_script as dynamic
import processStatic_script as static
import pickle
import scipy.io


# This script will parse out the battery cell parameters from the battery cell data obtained from
# "generate_battery_cell_data.py" python script. The parameters will be saved in the .pickle format.

data_origin = 'Typhoon'  # 'Boulder Colorado P14 battery cell data'
filename = 'P14_model.pickle'  # Name of the pickled file
# todo doHyst and numpoles variables are only tested when they are both one.
#  Extend this functionality to different values of these variables
numpoles = 1  # Number of resistor--capacitor pairs in final model
doHyst = 1    # Include hysteresis in model

# Initialize data
if data_origin == 'Boulder Colorado P14 battery cell data':  # test data from the Boulder university
    P14_DYN_50_P45 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P45.mat"), 45)
    P14_DYN_50_P25 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P25.mat"), 25)
    P14_DYN_30_P05 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_30_P05.mat"), 5)
    P14_OCV_P45 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P45.mat"), 45)
    P14_OCV_P25 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P25.mat"), 25)
    P14_OCV_P05 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P05.mat"), 5)
elif data_origin == 'Typhoon':  # Normal data format
    TYPHOON_FULL_CELL_DATA = data.CellAllData(scipy.io.loadmat("cell_all_data.mat"), [5, 25, 45], [5, 25, 45])

# Initialize model
cell_model = data.ESC_battery_model()

if data_origin == 'Boulder Colorado P14 battery cell data':
    # These next 2 functions require their data to have same temperature
    static.processStatic([P14_OCV_P05, P14_OCV_P25, P14_OCV_P45], cell_model)
    dynamic.processDynamic([P14_DYN_30_P05, P14_DYN_50_P25, P14_DYN_50_P45], cell_model, numpoles, doHyst)
elif data_origin == 'Typhoon':  # Normal data format
    # These next 2 functions require their data to have same temperature
    static.processStatic(TYPHOON_FULL_CELL_DATA['static_data'], cell_model)
    dynamic.processDynamic(TYPHOON_FULL_CELL_DATA['dynamic_data'], cell_model, numpoles, doHyst)

# Saving the model
print("Saving the model")
with open(filename, 'wb') as file:
    pickle.dump(cell_model, file)
    print("Model saved as ", filename)
