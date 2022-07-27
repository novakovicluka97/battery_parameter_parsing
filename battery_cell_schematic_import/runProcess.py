from battery_cell_data import *
from processDynamic_script import *
from processStatic_script import *
import pickle


# todo doHyst and numpoles variables are only tested when they are both one.
#  Extend this functionality to different values of these variables
numpoles = 1  # Number of resistor--capacitor pairs in final model
doHyst = 1    # Include hysteresis in model

# These next 2 functions require their data to have same temperature
processStatic([P14_OCV_P05, P14_OCV_P25, P14_OCV_P45], P14_model)
processDynamic([P14_DYN_30_P05, P14_DYN_50_P25, P14_DYN_50_P45], P14_model, numpoles, doHyst)

# Saving the model
print("Saving the model")
filename = 'P14_model.pickle'
with open(filename, 'wb') as file:
    pickle.dump(P14_model, file)
    ## here we will run the script for model object
    print("Model saved as ", filename)
