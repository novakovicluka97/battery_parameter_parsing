import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io
import typhoon.api.hil as hil
from typhoon.api.schematic_editor import model
import typhoon.test.reporting.messages as report
import typhoon.test.capture as capture
import typhoon.test.signals as signals
import battery_cell_data_functions as data
import pickle
from scipy.interpolate import interp1d
import numpy as np
import time


# This script will parse the general load current and tither profiles from Static and Dynamic scripts,
# from the original lab data from University of Boulder Colorado. Then it will collect the capture
# results from the Typhoon HIL capture. Right now, only VHIL is supported.

output_data_filename = 'cell_all_data.mat'
model_name = "Battery_parametrization_model.tse"
flag_show = False  # if True, certain graphs used for debugging will be shown
capture_duration = 2800  # default for VHIL is 2800, for HIL (30*60*60) seconds
capture_rate = 100  # 1 if not slowed down

# script directory
# Path to model file and to compiled model file
FILE_DIR_PATH = Path(__file__).parent
model_path = str(FILE_DIR_PATH / model_name)
compiled_model_path = model.get_compiled_model_file(model_path)


class Script:  # Format for the pickled cell data is this class per temperature, per script
    def __init__(self, time, temperature, voltage, current, chgAh, disAh):
        self.time = time
        self.temperature = temperature
        self.voltage = voltage
        self.current = current
        self.chgAh = chgAh
        self.disAh = disAh


if __name__ == "__main__":  # If this script is instantiated manually, recalculate current profiles
    # Loading the script data
    P14_OCV_P45 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P45.mat"), 45)
    P14_OCV_P25 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P25.mat"), 25)
    P14_OCV_P05 = data.OneTempStaticData(scipy.io.loadmat("P14_OCV_P05.mat"), 5)
    P14_DYN_50_P45 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P45.mat"), 45)
    P14_DYN_50_P25 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_50_P25.mat"), 25)
    P14_DYN_30_P05 = data.OneTempDynData(scipy.io.loadmat("P14_DYN_30_P05.mat"), 5)

    # Extracting the time vectors and resetting the x_axis to start with 0 for interpolator
    OCV_25_SCRIPT_2_TIME = P14_OCV_P25.script2.time - min(P14_OCV_P25.script2.time)
    OCV_25_SCRIPT_4_TIME = P14_OCV_P25.script4.time - min(P14_OCV_P25.script4.time)

    DYN_25_SCRIPT_1_TIME = P14_DYN_50_P25.script1.time - min(P14_DYN_50_P25.script1.time)

    # Extracting the current vectors
    OCV_25_SCRIPT_2_CURRENT = P14_OCV_P25.script2.current
    OCV_25_SCRIPT_4_CURRENT = P14_OCV_P25.script4.current

    DYN_25_SCRIPT_1_CURRENT = P14_DYN_50_P25.script1.current

    # Tither profiles extraction  # Todo: possibly better discharge tither profile at (12781 - 14585)
    TIME_TITHER_DISCHARGE = np.linspace(9370, 11170, 11170 - 9370 + 1)  # setting x_axis for tither profile LUT
    TITHER_DISCHARGE_interpolator = interp1d(OCV_25_SCRIPT_2_TIME, OCV_25_SCRIPT_2_CURRENT)  # interpolator
    CURRENT_TITHER_DISCHARGE = TITHER_DISCHARGE_interpolator(TIME_TITHER_DISCHARGE)  # y_axis for tither profile
    TIME_TITHER_DISCHARGE = TIME_TITHER_DISCHARGE - min(TIME_TITHER_DISCHARGE)  # redefining x_axis for tither profile
    TITHER_DISCHARGE_STOP_TIME = max(TIME_TITHER_DISCHARGE)

    TIME_TITHER_CHARGE = np.linspace(7440, 9030, 9030 - 7440 + 1)  # x_axis for tither profile start and finish
    TITHER_CHARGE_interpolator = interp1d(OCV_25_SCRIPT_4_TIME, OCV_25_SCRIPT_4_CURRENT)  # interpolator
    CURRENT_TITHER_CHARGE = TITHER_CHARGE_interpolator(TIME_TITHER_CHARGE)  # y_axis for charge tither profile
    TIME_TITHER_CHARGE = TIME_TITHER_CHARGE - min(TIME_TITHER_CHARGE)  # redefining x_axis for charge tither profile
    TITHER_CHARGE_STOP_TIME = max(TIME_TITHER_CHARGE)

    # Further modifications to current profiles
    CURRENT_TITHER_DISCHARGE = -(CURRENT_TITHER_DISCHARGE - 0.01)  # more dc-current to speed up profiles
    CURRENT_TITHER_CHARGE = (CURRENT_TITHER_CHARGE - 0.01)   # more dc-current to speed up profiles
    CURRENT_TITHER_DISCHARGE[-3:-1] = [0, 0]  # needed so internal resistance voltage drop can be calculated
    CURRENT_TITHER_CHARGE[-3:-1] = [0, 0]  # needed so internal resistance voltage drop can be calculated
    DYN_CURRENT_PROFILE = DYN_25_SCRIPT_1_CURRENT[1930:3380]  # extracting a single segment of this current profile
    DYN_TIME_PROFILE = DYN_25_SCRIPT_1_TIME[0:(3380-1930)]
    DYN_PROFILE_STOP_TIME = DYN_25_SCRIPT_1_TIME[(3380-1930)]

    current_profiles_dict = {  # Todo: reapply the names to this dictionary and schematic in next commit
        'OCV_25_SCRIPT_2_TIME_TITHER': TIME_TITHER_DISCHARGE,
        'OCV_25_SCRIPT_4_TIME_TITHER': TIME_TITHER_CHARGE,
        'OCV_25_SCRIPT_2_CURRENT_TITHER': CURRENT_TITHER_DISCHARGE,
        'OCV_25_SCRIPT_4_CURRENT_TITHER': CURRENT_TITHER_CHARGE,
        'OCV_25_SCRIPT_2_TIME_STOP': TITHER_DISCHARGE_STOP_TIME,
        'OCV_25_SCRIPT_4_TIME_STOP': TITHER_CHARGE_STOP_TIME,

        'DYN_25_SCRIPT_1_CURRENT': DYN_CURRENT_PROFILE,  # mean is 1.981
        'DYN_25_SCRIPT_1_TIME': DYN_TIME_PROFILE,
        'DYN_SCRIPT_1_STOP': DYN_PROFILE_STOP_TIME
    }

    print("Saving the current load profiles that will be injected into batteries via LUTs")
    filename = 'current_profiles.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(current_profiles_dict, file)
        print("Current profiles saved as: ", filename)

    # data.plot_func([OCV_25_SCRIPT_2_TIME_TITHER, OCV_25_SCRIPT_4_TIME_TITHER],
    #                [OCV_25_SCRIPT_2_CURRENT_TITHER, OCV_25_SCRIPT_4_CURRENT_TITHER],
    #                ["OCV_25_SCRIPT_2_CURRENT_TITHER", "OCV_25_SCRIPT_4_CURRENT_TITHER"],
    #                flag_show=flag_show)

# Run very fast simulations and thus to obtain the data quickly before the physical tests take place
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

# signals for capturing
channel_signals = ["temperature_1", "voltage_1", "current_1", "chgAh_1", "disAh_1", "script_no_1",
                   "temperature_2", "voltage_2", "current_2", "chgAh_2", "disAh_2", "script_no_2",
                   "temperature_3", "voltage_3", "current_3", "chgAh_3", "disAh_3", "script_no_3",
                   "dyn_temperature_1", "dyn_voltage_1", "dyn_current_1", "dyn_chgAh_1", "dyn_disAh_1", "dyn_script_no_1",
                   "dyn_temperature_2", "dyn_voltage_2", "dyn_current_2", "dyn_chgAh_2", "dyn_disAh_2", "dyn_script_no_2",
                   "dyn_temperature_3", "dyn_voltage_3", "dyn_current_3", "dyn_chgAh_3", "dyn_disAh_3", "dyn_script_no_3",
                   "Time", "done_flag"]

capture.start_capture(duration=capture_duration,
                      rate=capture_rate,
                      signals=channel_signals,
                      executeAt=0.0)

hil.start_simulation()
print('STEP 3: Starting simulation and capture')
if vhil_device:
    print('     This may take up to ' + str(round(capture_duration/10)) + ' seconds')
else:
    print('     This may take up to ' + str(round(capture_duration)) + ' seconds')
st = time.time()  # measuring the starting time

# capture.wait_until("done_flag", 'above', 0.5, timeout=capture_duration)
capture.wait_capture_finish()

et = time.time()  # get the end time
elapsed_time = et - st  # get the execution time
print('Execution time:', elapsed_time, 'seconds, and ', hil.get_sim_time(), "seconds of simulation time")
cap_data = capture.get_capture_results()
hil.stop_simulation()  # Stopping the simulation

print('STEP 4: Manipulating the captured data')
# Assigning to variables
time_vec = list(cap_data["Time"])

# STATIC TEST
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

# data.plot_func([time_vec], [voltage_3], ["Script4_05"], flag_show=True)

# DYNAMIC TESTS
dyn_current_1       = list(cap_data["dyn_current_1"])
dyn_voltage_1       = list(cap_data["dyn_voltage_1"])
dyn_script_no_1     = list(cap_data["dyn_script_no_1"])
dyn_temperature_1   = list(cap_data["dyn_temperature_1"])
dyn_chgAh_1         = list(cap_data["dyn_chgAh_1"])
dyn_disAh_1         = list(cap_data["dyn_disAh_1"])

dyn_current_2       = list(cap_data["dyn_current_2"])
dyn_voltage_2       = list(cap_data["dyn_voltage_2"])
dyn_script_no_2     = list(cap_data["dyn_script_no_2"])
dyn_temperature_2   = list(cap_data["dyn_temperature_2"])
dyn_chgAh_2         = list(cap_data["dyn_chgAh_2"])
dyn_disAh_2         = list(cap_data["dyn_disAh_2"])

dyn_current_3       = list(cap_data["dyn_current_3"])
dyn_voltage_3       = list(cap_data["dyn_voltage_3"])
dyn_script_no_3     = list(cap_data["dyn_script_no_3"])
dyn_temperature_3   = list(cap_data["dyn_temperature_3"])
dyn_chgAh_3         = list(cap_data["dyn_chgAh_3"])
dyn_disAh_3         = list(cap_data["dyn_disAh_3"])

# Parsing the data
dyn_scr_start_2_25 = dyn_script_no_1.index(2)
dyn_scr_start_3_25 = dyn_script_no_1.index(3)
dyn_scr_start_4_25 = dyn_script_no_1.index(4)

dyn_scr_start_2_45 = dyn_script_no_2.index(2)
dyn_scr_start_3_45 = dyn_script_no_2.index(3)
dyn_scr_start_4_45 = dyn_script_no_2.index(4)

dyn_scr_start_2_05 = dyn_script_no_3.index(2)
dyn_scr_start_3_05 = dyn_script_no_3.index(3)
dyn_scr_start_4_05 = dyn_script_no_3.index(4)

# Packing the data into class script
# TEMPERATURE 25 DYNAMIC TEST
Dyn_Script1_25 = Script(time_vec[0:dyn_scr_start_2_25], dyn_temperature_1[0:dyn_scr_start_2_25], dyn_voltage_1[0:dyn_scr_start_2_25],
                    dyn_current_1[0:dyn_scr_start_2_25], dyn_chgAh_1[0:dyn_scr_start_2_25], dyn_disAh_1[0:dyn_scr_start_2_25])
Dyn_Script2_25 = Script(time_vec[dyn_scr_start_2_25:dyn_scr_start_3_25], dyn_temperature_1[dyn_scr_start_2_25:dyn_scr_start_3_25],
                    dyn_voltage_1[dyn_scr_start_2_25:dyn_scr_start_3_25], dyn_current_1[dyn_scr_start_2_25:dyn_scr_start_3_25],
                    dyn_chgAh_1[dyn_scr_start_2_25:dyn_scr_start_3_25], dyn_disAh_1[dyn_scr_start_2_25:dyn_scr_start_3_25])
Dyn_Script3_25 = Script(time_vec[dyn_scr_start_3_25:dyn_scr_start_4_25], dyn_temperature_1[dyn_scr_start_3_25:dyn_scr_start_4_25],
                    dyn_voltage_1[dyn_scr_start_3_25:dyn_scr_start_4_25], dyn_current_1[dyn_scr_start_3_25:dyn_scr_start_4_25],
                    dyn_chgAh_1[dyn_scr_start_3_25:dyn_scr_start_4_25], dyn_disAh_1[dyn_scr_start_3_25:dyn_scr_start_4_25])

# TEMPERATURE 45 DYNAMIC TEST
Dyn_Script1_45 = Script(time_vec[0:dyn_scr_start_2_45], dyn_temperature_2[0:dyn_scr_start_2_45], dyn_voltage_2[0:dyn_scr_start_2_45],
                    dyn_current_2[0:dyn_scr_start_2_45], dyn_chgAh_2[0:dyn_scr_start_2_45], dyn_disAh_2[0:dyn_scr_start_2_45])
Dyn_Script2_45 = Script(time_vec[dyn_scr_start_2_45:dyn_scr_start_3_45], dyn_temperature_2[dyn_scr_start_2_45:dyn_scr_start_3_45],
                    dyn_voltage_2[dyn_scr_start_2_45:dyn_scr_start_3_45], dyn_current_2[dyn_scr_start_2_45:dyn_scr_start_3_45],
                    dyn_chgAh_2[dyn_scr_start_2_45:dyn_scr_start_3_45], dyn_disAh_2[dyn_scr_start_2_45:dyn_scr_start_3_45])
Dyn_Script3_45 = Script(time_vec[dyn_scr_start_3_45:dyn_scr_start_4_45], dyn_temperature_2[dyn_scr_start_3_45:dyn_scr_start_4_45],
                    dyn_voltage_2[dyn_scr_start_3_45:dyn_scr_start_4_45], dyn_current_2[dyn_scr_start_3_45:dyn_scr_start_4_45],
                    dyn_chgAh_2[dyn_scr_start_3_45:dyn_scr_start_4_45], dyn_disAh_2[dyn_scr_start_3_45:dyn_scr_start_4_45])

# TEMPERATURE 05 DYNAMIC TEST
Dyn_Script1_05 = Script(time_vec[0:dyn_scr_start_2_05], dyn_temperature_3[0:dyn_scr_start_2_05], dyn_voltage_3[0:dyn_scr_start_2_05],
                    dyn_current_3[0:dyn_scr_start_2_05], dyn_chgAh_3[0:dyn_scr_start_2_05], dyn_disAh_3[0:dyn_scr_start_2_05])
Dyn_Script2_05 = Script(time_vec[dyn_scr_start_2_05:dyn_scr_start_3_05], dyn_temperature_3[dyn_scr_start_2_05:dyn_scr_start_3_05],
                    dyn_voltage_3[dyn_scr_start_2_05:dyn_scr_start_3_05], dyn_current_3[dyn_scr_start_2_05:dyn_scr_start_3_05],
                    dyn_chgAh_3[dyn_scr_start_2_05:dyn_scr_start_3_05], dyn_disAh_3[dyn_scr_start_2_05:dyn_scr_start_3_05])
Dyn_Script3_05 = Script(time_vec[dyn_scr_start_3_05:dyn_scr_start_4_05], dyn_temperature_3[dyn_scr_start_3_05:dyn_scr_start_4_05],
                    dyn_voltage_3[dyn_scr_start_3_05:dyn_scr_start_4_05], dyn_current_3[dyn_scr_start_3_05:dyn_scr_start_4_05],
                    dyn_chgAh_3[dyn_scr_start_3_05:dyn_scr_start_4_05], dyn_disAh_3[dyn_scr_start_3_05:dyn_scr_start_4_05])

# Saving the data
data = {
    "OCVData_25": [Script1_25, Script2_25, Script3_25, Script4_25],
    "OCVData_45": [Script1_45, Script2_45, Script3_45, Script4_45],
    "OCVData_05": [Script1_05, Script2_05, Script3_05, Script4_05],
    "DYNData_25": [Dyn_Script1_25, Dyn_Script2_25, Dyn_Script3_25],
    "DYNData_45": [Dyn_Script1_45, Dyn_Script2_45, Dyn_Script3_45],
    "DYNData_05": [Dyn_Script1_05, Dyn_Script2_05, Dyn_Script3_05],
}

print('STEP 5: Saving the data into a file: ' + output_data_filename)
scipy.io.savemat(output_data_filename, data)
