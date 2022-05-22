from battery_cell_data import *
from processDynamic_script import *
from processStatic_script import *


# setupDynData is a function that specifies the data for the battery: P14 and temps 30_05, 50_25, 50_45
numpoles = 1  # number of resistor--capacitor pairs in final model
doHyst = 1    # whether to include hysteresis in model

processStatic([P14_OCV_P05, P14_OCV_P25, P14_OCV_P45], P14_model)
processDynamic([P14_DYN_30_P05, P14_DYN_50_P25, P14_DYN_50_P45], P14_model, numpoles, doHyst)
print("Done!")