import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io
import typhoon.api.hil as hil
from typhoon.api.schematic_editor import model
import typhoon.test.reporting.messages as report
import typhoon.test.capture as capture
import battery_cell_data_functions as data
import pickle
from scipy.interpolate import interp1d
import numpy as np
import time


# This script will parse the general load current and tither profiles for Static and Dynamic scripts,
# from the original lab data from University of Boulder Colorado. Then it will collect the capture
# results from the Typhoon HIL measurements. Right now, only VHIL is supported.

output_data_filename = 'cell_all_data.mat'
model_name = "Battery_parametrization_model.tse"
flag_show = True  # if True, certain graphs used for debugging will be shown
capture_duration = 30 * 60 * 60

# script directory
# Path to model file and to compiled model file
FILE_DIR_PATH = Path(__file__).parent

model_path = str(FILE_DIR_PATH / model_name)
compiled_model_path = model.get_compiled_model_file(model_path)


class Script:
    def __init__(self, time, temperature, voltage, current, chgAh, disAh):
        self.time = time
        self.temperature = temperature
        self.voltage = voltage
        self.current = current
        self.chgAh = chgAh
        self.disAh = disAh


if __name__ == "__main__":  # If this script is instantiated manually...
    # Loads the script data
    P14_DYN_50_P45 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P45.mat"), 45)
    P14_DYN_50_P25 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P25.mat"), 25)
    P14_DYN_30_P05 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_30_P05.mat"), 5)
    P14_OCV_P45 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P45.mat"), 45)
    P14_OCV_P25 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P25.mat"), 25)
    P14_OCV_P05 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P05.mat"), 5)

    # Refactors and loads the time vector (it starts at a certain point)
    DYN_25_SCRIPT_1_TIME = P14_DYN_50_P25.script1.time - min(P14_DYN_50_P25.script1.time)
    DYN_25_SCRIPT_2_TIME = P14_DYN_50_P25.script2.time - min(P14_DYN_50_P25.script2.time)
    DYN_25_SCRIPT_3_TIME = P14_DYN_50_P25.script3.time - min(P14_DYN_50_P25.script3.time)

    OCV_25_SCRIPT_2_TIME = P14_OCV_P25.script2.time - min(P14_OCV_P25.script2.time)
    OCV_25_SCRIPT_4_TIME = P14_OCV_P25.script4.time - min(P14_OCV_P25.script4.time)

    # Loads the current vectors
    DYN_25_SCRIPT_1_CURRENT = P14_DYN_50_P25.script1.current
    DYN_25_SCRIPT_2_CURRENT = P14_DYN_50_P25.script2.current
    DYN_25_SCRIPT_3_CURRENT = P14_DYN_50_P25.script3.current

    OCV_25_SCRIPT_2_CURRENT = P14_OCV_P25.script2.current
    OCV_25_SCRIPT_4_CURRENT = P14_OCV_P25.script4.current

    # Tither profiles extraction
    OCV_25_SCRIPT_2_interpolator = interp1d(OCV_25_SCRIPT_2_TIME, OCV_25_SCRIPT_2_CURRENT)
    OCV_25_SCRIPT_2_TIME_TITHER = np.linspace(9370, 11170,
                                              11170 - 9370 + 1)  # set points for tither profile start and finish
    OCV_25_SCRIPT_2_CURRENT_TITHER = OCV_25_SCRIPT_2_interpolator(OCV_25_SCRIPT_2_TIME_TITHER)
    OCV_25_SCRIPT_2_TIME_TITHER = OCV_25_SCRIPT_2_TIME_TITHER - min(OCV_25_SCRIPT_2_TIME_TITHER)
    OCV_25_SCRIPT_2_TIME_STOP = max(OCV_25_SCRIPT_2_TIME_TITHER)

    OCV_25_SCRIPT_4_interpolator = interp1d(OCV_25_SCRIPT_4_TIME, OCV_25_SCRIPT_4_CURRENT)
    OCV_25_SCRIPT_4_TIME_TITHER = np.linspace(7440, 9030,
                                              9030 - 7440 + 1)  # set points for tither profile start and finish
    OCV_25_SCRIPT_4_CURRENT_TITHER = OCV_25_SCRIPT_4_interpolator(OCV_25_SCRIPT_4_TIME_TITHER)
    OCV_25_SCRIPT_4_TIME_TITHER = OCV_25_SCRIPT_4_TIME_TITHER - min(OCV_25_SCRIPT_4_TIME_TITHER)
    OCV_25_SCRIPT_4_TIME_STOP = max(OCV_25_SCRIPT_4_TIME_TITHER)

    data.plot_func([OCV_25_SCRIPT_2_TIME_TITHER, OCV_25_SCRIPT_4_TIME_TITHER],
                   [OCV_25_SCRIPT_2_CURRENT_TITHER, OCV_25_SCRIPT_4_CURRENT_TITHER],
                   ["OCV_25_SCRIPT_2_CURRENT_TITHER", "OCV_25_SCRIPT_4_CURRENT_TITHER"],
                   flag_show=flag_show)
    data.plot_func([DYN_25_SCRIPT_1_TIME, DYN_25_SCRIPT_2_TIME, DYN_25_SCRIPT_3_TIME],
                   [DYN_25_SCRIPT_1_CURRENT, DYN_25_SCRIPT_2_CURRENT, DYN_25_SCRIPT_3_CURRENT],
                   ["DYN_25_SCRIPT_1_CURRENT", "DYN_25_SCRIPT_2_CURRENT", "DYN_25_SCRIPT_3_CURRENT"],
                   flag_show=flag_show)
    data.plot_func([P14_OCV_P25.script1.time, P14_OCV_P25.script2.time, P14_OCV_P25.script3.time, P14_OCV_P25.script4.time],
                   [P14_OCV_P25.script1.current, P14_OCV_P25.script2.current, P14_OCV_P25.script3.current, P14_OCV_P25.script4.current],
                   ["P14_OCV_P25.script1.current", "P14_OCV_P25.script2.current", "P14_OCV_P25.script3.current", "P14_OCV_P25.script4.current"],
                   flag_show=flag_show)
    data.plot_func([P14_OCV_P25.script1.time, P14_OCV_P25.script2.time, P14_OCV_P25.script3.time, P14_OCV_P25.script4.time],
                   [P14_OCV_P25.script1.voltage, P14_OCV_P25.script2.voltage, P14_OCV_P25.script3.voltage, P14_OCV_P25.script4.voltage],
                   ["P14_OCV_P25.script1.current", "P14_OCV_P25.script2.current", "P14_OCV_P25.script3.current", "P14_OCV_P25.script4.current"],
                   flag_show=flag_show)

    current_profiles_dict = {
        'OCV_25_SCRIPT_2_TIME_TITHER': OCV_25_SCRIPT_2_TIME_TITHER,
        'OCV_25_SCRIPT_4_TIME_TITHER': OCV_25_SCRIPT_4_TIME_TITHER,
        'OCV_25_SCRIPT_2_CURRENT_TITHER': -(OCV_25_SCRIPT_2_CURRENT_TITHER - 0.01),
        'OCV_25_SCRIPT_4_CURRENT_TITHER': (OCV_25_SCRIPT_4_CURRENT_TITHER - 0.01),  # This is just because i sampled
        # the tither signal in a way that it is still negative in average so I omitted the minus sign
        'OCV_25_SCRIPT_2_TIME_STOP': OCV_25_SCRIPT_2_TIME_STOP,
        'OCV_25_SCRIPT_4_TIME_STOP': OCV_25_SCRIPT_4_TIME_STOP,
    }

    print("Saving the current load profiles that will be injected into batteries")
    filename = 'current_profiles.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(current_profiles_dict, file)
        print("Current profiles saved as: ", filename)

# Use VHIL to run very fast simulations and thus to obtain the data quickly before the physical tests take place
vhil_device = True
model.load(model_path)
print('STEP 1: Initializing the model')
model_device = model.get_model_property_value("hil_device")
model_config = model.get_model_property_value("hil_configuration_id")
report.report_message(
    "Virtual HIL device is used. Model is compiled for {} C{}.".format(model_device, model_config))

model.compile()
print('STEP 2: Compiling and loading the model')
hil.load_model(compiled_model_path, vhil_device=vhil_device)

battery_cell_name = "Battery Cell_25"
# signals for capturing
channel_signals = ["temperature_1", "voltage_1", "current_1", "chgAh_1", "disAh_1", "script_no_1",
                   "temperature_2", "voltage_2", "current_2", "chgAh_2", "disAh_2", "script_no_2",
                   "temperature_3", "voltage_3", "current_3", "chgAh_3", "disAh_3", "script_no_3"]

capture.start_capture(duration=capture_duration,
                      rate=256,  # Todo try lower rate
                      signals=channel_signals,
                      executeAt=0.0)

hil.start_simulation()
print('STEP 3: Starting simulation and capture')
print('     This may take up to an hour')
st = time.time()  # measuring the starting time

# Todo make the capture stop once the signal reaches a certain value
capture.wait_until("script_no_1", 'above', 4.5, timeout=capture_duration)
capture.wait(100)  # Wait 100 more seconds

et = time.time()  # get the end time
elapsed_time = et - st  # get the execution time
print('Execution time:', elapsed_time, 'seconds, and ', hil.get_sim_time(), "seconds of simulation time")
cap_data = capture.get_capture_results()

hil.stop_simulation()  # Stopping the simulation

print('STEP 4: Manipulating the captured data')
time_vec = []
for i in range(len(cap_data.T)):  # converting data-series to time
    time_vec.append(list(cap_data.T)[i].total_seconds())

# Assigning to variables
current_1 = list(cap_data["current_1"])
voltage_1 = list(cap_data["voltage_1"])
script_no_1 = list(cap_data["script_no_1"])
temperature_1 = list(cap_data["temperature_1"])
chgAh_1 = list(cap_data["chgAh_1"])
disAh_1 = list(cap_data["disAh_1"])

current_2 = list(cap_data["current_2"])
voltage_2 = list(cap_data["voltage_2"])
script_no_2 = list(cap_data["script_no_2"])
temperature_2 = list(cap_data["temperature_2"])
chgAh_2 = list(cap_data["chgAh_2"])
disAh_2 = list(cap_data["disAh_2"])

current_3 = list(cap_data["current_3"])
voltage_3 = list(cap_data["voltage_3"])
script_no_3 = list(cap_data["script_no_3"])
temperature_3 = list(cap_data["temperature_3"])
chgAh_3 = list(cap_data["chgAh_3"])
disAh_3 = list(cap_data["disAh_3"])

# Parsing the data
scr_start_2_25 = script_no_1.index(2)
scr_start_3_25 = script_no_1.index(3)
scr_start_4_25 = script_no_1.index(4)
scr_start_5_25 = script_no_1.index(5)

scr_start_2_45 = script_no_2.index(2)
scr_start_3_45 = script_no_2.index(3)
scr_start_4_45 = script_no_2.index(4)
scr_start_5_45 = script_no_2.index(5)

scr_start_2_05 = script_no_3.index(2)
scr_start_3_05 = script_no_3.index(3)
scr_start_4_05 = script_no_3.index(4)
scr_start_5_05 = script_no_3.index(5)

# Packing the data into class script
# TEMPERATURE 25 STATIC TEST
Script1_25 = Script(time_vec[0:scr_start_2_25], temperature_1[0:scr_start_2_25], voltage_1[0:scr_start_2_25],
                    current_1[0:scr_start_2_25], chgAh_1[0:scr_start_2_25], disAh_1[0:scr_start_2_25])
Script2_25 = Script(time_vec[scr_start_2_25:scr_start_3_25], temperature_1[scr_start_2_25:scr_start_3_25],
                    voltage_1[scr_start_2_25:scr_start_3_25], current_1[scr_start_2_25:scr_start_3_25],
                    chgAh_1[scr_start_2_25:scr_start_3_25], disAh_1[scr_start_2_25:scr_start_3_25])
Script3_25 = Script(time_vec[scr_start_3_25:scr_start_4_25], temperature_1[scr_start_3_25:scr_start_4_25],
                    voltage_1[scr_start_3_25:scr_start_4_25], current_1[scr_start_3_25:scr_start_4_25],
                    chgAh_1[scr_start_3_25:scr_start_4_25], disAh_1[scr_start_3_25:scr_start_4_25])
Script4_25 = Script(time_vec[scr_start_4_25:scr_start_5_25], temperature_1[scr_start_4_25:scr_start_5_25],
                    voltage_1[scr_start_4_25:scr_start_5_25], current_1[scr_start_4_25:scr_start_5_25],
                    chgAh_1[scr_start_4_25:scr_start_5_25], disAh_1[scr_start_4_25:scr_start_5_25])

# TEMPERATURE 45 STATIC TEST
Script1_45 = Script(time_vec[0:scr_start_2_45], temperature_2[0:scr_start_2_45], voltage_2[0:scr_start_2_45],
                    current_2[0:scr_start_2_45], chgAh_2[0:scr_start_2_45], disAh_2[0:scr_start_2_45])
Script2_45 = Script(time_vec[scr_start_2_45:scr_start_3_45], temperature_2[scr_start_2_45:scr_start_3_45],
                    voltage_2[scr_start_2_45:scr_start_3_45], current_2[scr_start_2_45:scr_start_3_45],
                    chgAh_2[scr_start_2_45:scr_start_3_45], disAh_2[scr_start_2_45:scr_start_3_45])
Script3_45 = Script(time_vec[scr_start_3_45:scr_start_4_45], temperature_2[scr_start_3_45:scr_start_4_45],
                    voltage_2[scr_start_3_45:scr_start_4_45], current_2[scr_start_3_45:scr_start_4_45],
                    chgAh_2[scr_start_3_45:scr_start_4_45], disAh_2[scr_start_3_45:scr_start_4_45])
Script4_45 = Script(time_vec[scr_start_4_45:scr_start_5_45], temperature_2[scr_start_4_45:scr_start_5_45],
                    voltage_2[scr_start_4_45:scr_start_5_45], current_2[scr_start_4_45:scr_start_5_45],
                    chgAh_2[scr_start_4_45:scr_start_5_45], disAh_2[scr_start_4_45:scr_start_5_45])

# TEMPERATURE 05 STATIC TEST
Script1_05 = Script(time_vec[0:scr_start_2_05], temperature_3[0:scr_start_2_05], voltage_3[0:scr_start_2_05],
                    current_3[0:scr_start_2_05], chgAh_3[0:scr_start_2_05], disAh_3[0:scr_start_2_05])
Script2_05 = Script(time_vec[scr_start_2_05:scr_start_3_05], temperature_3[scr_start_2_05:scr_start_3_05],
                    voltage_3[scr_start_2_05:scr_start_3_05], current_3[scr_start_2_05:scr_start_3_05],
                    chgAh_3[scr_start_2_05:scr_start_3_05], disAh_3[scr_start_2_05:scr_start_3_05])
Script3_05 = Script(time_vec[scr_start_3_05:scr_start_4_05], temperature_3[scr_start_3_05:scr_start_4_05],
                    voltage_3[scr_start_3_05:scr_start_4_05], current_3[scr_start_3_05:scr_start_4_05],
                    chgAh_3[scr_start_3_05:scr_start_4_05], disAh_3[scr_start_3_05:scr_start_4_05])
Script4_05 = Script(time_vec[scr_start_4_05:scr_start_5_05], temperature_3[scr_start_4_05:scr_start_5_05],
                    voltage_3[scr_start_4_05:scr_start_5_05], current_3[scr_start_4_05:scr_start_5_05],
                    chgAh_3[scr_start_4_05:scr_start_5_05], disAh_3[scr_start_4_05:scr_start_5_05])

data.plot_func([time_vec], [voltage_3], ["Script4_05"], flag_show=True)








# Saving the data
data = {
    "OCVData_25": [Script1_25, Script2_25, Script3_25, Script4_25],
    "OCVData_05": [Script1_05, Script2_05, Script3_05, Script4_05],
    "OCVData_45": [Script1_45, Script2_45, Script3_45, Script4_45],
    "DYNData_25": [Script1_25, Script1_25, Script1_25],
    "DYNData_05": [Script1_25, Script1_25, Script1_25],
    "DYNData_45": [Script1_25, Script1_25, Script1_25],
}

print('STEP 5: Saving the data into a file: ' + output_data_filename)
scipy.io.savemat(output_data_filename, data)
