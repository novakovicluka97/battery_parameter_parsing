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

# Parameters:
test_temperatures = ['25', '45', '5']  # must exist in the model as well
Ts = 1e-2  # signal processing execution rate
capture_duration = 28e2  # default for VHIL is 28e2, for HIL (30*60*60) seconds, for VHIL full no hyst and no RC 23e4
SIMULATION_SPEED_UP = 100  # 1 if not slowed down
M0 = [0.0031315, 0.0023535, 0.0011502]
M = [0.039929, 0.020018, 0.020545]
M0 = [1e-5, 1e-5, 1e-5]
M = [1e-5, 1e-5, 1e-5]
R1 = 0.64769e-3
C1 = 7.9598e3

capture_rate = SIMULATION_SPEED_UP
output_data_filename = 'Typhoon_captured_data.mat'
model_name = "Battery_parametrization_model.tse"  # "Battery_parametrization_model.tse"
flag_show = False  # if True, certain graphs used for debugging will be shown
current_profile_filename = 'current_profiles.pickle'

# script directory
# Path to model file and to compiled model file
FILE_DIR_PATH = Path(__file__).parent
model_path = str(FILE_DIR_PATH / model_name)
compiled_model_path = model.get_compiled_model_file(model_path)

model_init_code = f"""
R1 = {str(R1)}
C1 = {str(C1)}
M0 = {str(M0)}
M = {str(M)}
SIMULATION_SPEED_UP = {str(SIMULATION_SPEED_UP)}
init_dynamic = 100  # 90% initial state of charge for dynamic scripts so that the voltage doesnt go over Vmax
Ts = {str(Ts)}

rest_time_for_temperature_equilibrium = 7200/SIMULATION_SPEED_UP  # 7200s or 2h
counter_cooldown_max = rest_time_for_temperature_equilibrium/Ts
counter_cooldown_max = counter_cooldown_max/SIMULATION_SPEED_UP

Vmax = 4.1497   # Volts
Vmin = 2.8060   # Volts
total_Q_original = [14.592, 14.532, 14.444]
total_Q = [i/SIMULATION_SPEED_UP for i in total_Q_original]
G = [67.207, 92.645, 67.840]  # G param is wrong for default battery

C = 14.532  # C rate of a battery is current that will empty the cell in one hour
DISCHG_RATE =  0.5 # 0.5 if its normal battery discharg speed
CHG_RATE = -DISCHG_RATE

# when the dynamic script executes, current profiles don't push the cell voltage
# over the intended Vmax
V_end_script_1 = (Vmax-Vmin)*0.1+Vmin  # ends dynamic script 1 when SOC is roughly at 10%
DYN_DIS_CHG = C/30
DYN_CHG = -C

import pickle
# model_path = mdl.get_model_file_path()
filename = 'c:\\\\PROJECT\\\\battery_cell_testing\\\\scripts_and_data\\\\{current_profile_filename}'
file = open(filename, 'rb')
current_profiles_dict = pickle.load(file)

TIME_TITHER_DISCHARGE      = current_profiles_dict['TIME_TITHER_DISCHARGE']
TIME_TITHER_CHARGE         = current_profiles_dict['TIME_TITHER_CHARGE']
CURRENT_TITHER_DISCHARGE   = current_profiles_dict['CURRENT_TITHER_DISCHARGE']
CURRENT_TITHER_CHARGE      = current_profiles_dict['CURRENT_TITHER_CHARGE']
TITHER_DISCHARGE_STOP_TIME = current_profiles_dict['TITHER_DISCHARGE_STOP_TIME']
TITHER_CHARGE_STOP_TIME    = current_profiles_dict['TITHER_CHARGE_STOP_TIME']

DYN_TIME_PROFILE           = current_profiles_dict['DYN_TIME_PROFILE']
DYN_CURRENT_PROFILE        = current_profiles_dict['DYN_CURRENT_PROFILE']
DYN_PROFILE_STOP_TIME      = current_profiles_dict['DYN_PROFILE_STOP_TIME']

# Tither profiles are assumed to be the same for dynamic scripts as well as for 
# the static scripts, check later

TITHER_DISCHARGE_STOP_TIME       = TITHER_DISCHARGE_STOP_TIME /SIMULATION_SPEED_UP
TITHER_CHARGE_STOP_TIME          = TITHER_CHARGE_STOP_TIME    /SIMULATION_SPEED_UP
DYN_PROFILE_STOP_TIME            = DYN_PROFILE_STOP_TIME      /SIMULATION_SPEED_UP

"""


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

    # Current profiles extraction  # Todo: possibly better discharge tither profile at (12781 - 14585)
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

    DYN_CURRENT_PROFILE = DYN_25_SCRIPT_1_CURRENT[1930:3380]  # extracting a single segment of this current profile
    DYN_TIME_PROFILE = DYN_25_SCRIPT_1_TIME[0:(3380-1930)]
    DYN_PROFILE_STOP_TIME = DYN_25_SCRIPT_1_TIME[(3380-1930)]

    # Further modifications to current profiles
    CURRENT_TITHER_DISCHARGE = -(CURRENT_TITHER_DISCHARGE - 0.01)  # more dc-current to speed up profiles
    CURRENT_TITHER_CHARGE = (CURRENT_TITHER_CHARGE - 0.01)   # more dc-current to speed up profiles
    CURRENT_TITHER_DISCHARGE[-3:-1] = [0, 0]  # needed so internal resistance voltage drop can be calculated
    CURRENT_TITHER_CHARGE[-3:-1] = [0, 0]  # needed so internal resistance voltage drop can be calculated
    DYN_CURRENT_PROFILE[-1] = 0
    DYN_CURRENT_PROFILE[0] = 0

    current_profiles_dict = {
        'TIME_TITHER_DISCHARGE': TIME_TITHER_DISCHARGE,
        'TIME_TITHER_CHARGE': TIME_TITHER_CHARGE,
        'CURRENT_TITHER_DISCHARGE': CURRENT_TITHER_DISCHARGE,
        'CURRENT_TITHER_CHARGE': CURRENT_TITHER_CHARGE,
        'TITHER_DISCHARGE_STOP_TIME': TITHER_DISCHARGE_STOP_TIME,
        'TITHER_CHARGE_STOP_TIME': TITHER_CHARGE_STOP_TIME,

        'DYN_CURRENT_PROFILE': DYN_CURRENT_PROFILE,  # mean is 1.981
        'DYN_TIME_PROFILE': DYN_TIME_PROFILE,
        'DYN_PROFILE_STOP_TIME': DYN_PROFILE_STOP_TIME
    }

    print("Saving the current load profiles that will be injected into batteries via LUTs")
    with open(current_profile_filename, 'wb') as file:
        pickle.dump(current_profiles_dict, file)
        print("Current profiles saved as: ", current_profile_filename)

print('STEP 1: Initializing the model')
# Run very fast simulations and thus to obtain the data quickly before the physical tests take place
vhil_device = True
model.load(model_path)
model_device = model.get_model_property_value("hil_device")
model_config = model.get_model_property_value("hil_configuration_id")
report.report_message("Virtual HIL device is used. Model is compiled for {} C{}.".format(model_device, model_config))
model.set_model_init_code(model_init_code)
model.save()  # saving a model

print('STEP 2: Compiling and loading the model, vhil_device = ', vhil_device)
if model.compile():
    print("     Compile successful.")
else:
    print("     Compile failed.")
model.close_model()

hil.load_model(compiled_model_path, vhil_device=vhil_device)

# signals for capturing
channel_signals = ['Time', 'done_flag']
for temp in test_temperatures:
    for measurement in ['temperature', 'voltage', 'current', 'chgAh', 'disAh', 'script_no']:
        channel_signals.append('static_' + temp + '.' + measurement)
        channel_signals.append('dynamic_' + temp + '.' + measurement)

capture.start_capture(duration=capture_duration,
                      rate=capture_rate,
                      signals=channel_signals,
                      executeAt=0.0)

print('STEP 3: Starting simulation and capture')
hil.start_simulation()
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

print('STEP 4: Parsing the captured data and saving it in a dictionary')
time_vec = list(cap_data["Time"])
data = dict()  # creating an output dictionary for all the dynamic and static data

for temp in test_temperatures:
    for script_type in ['static', 'dynamic']:
        current = list(cap_data[script_type + '_' + temp + '.current'])
        voltage = list(cap_data[script_type + '_' + temp + '.voltage'])
        script_no = list(cap_data[script_type + '_' + temp + '.script_no'])
        temperature = list(cap_data[script_type + '_' + temp + '.temperature'])
        chgAh = list(cap_data[script_type + '_' + temp + '.chgAh']/3600)  # converting to Ah from As
        disAh = list(cap_data[script_type + '_' + temp + '.disAh']/3600)  # converting to Ah from As

        script_1_stop = script_no.index(2)
        script_2_stop = script_no.index(3)
        script_3_stop = script_no.index(4)

        # Create script objects for each part of the simulation. Lists are segmented using script_x_stop variables
        Script_1 = Script(time_vec[0:script_1_stop], temperature[0:script_1_stop], voltage[0:script_1_stop],
                          current[0:script_1_stop], chgAh[0:script_1_stop], disAh[0:script_1_stop])
        Script_2 = Script(time_vec[script_1_stop:script_2_stop], temperature[script_1_stop:script_2_stop],
                          voltage[script_1_stop:script_2_stop],
                          current[script_1_stop:script_2_stop], chgAh[script_1_stop:script_2_stop],
                          disAh[script_1_stop:script_2_stop])
        Script_3 = Script(time_vec[script_2_stop:script_3_stop], temperature[script_2_stop:script_3_stop],
                          voltage[script_2_stop:script_3_stop],
                          current[script_2_stop:script_3_stop], chgAh[script_2_stop:script_3_stop],
                          disAh[script_2_stop:script_3_stop])

        if script_type == 'static':  # add new Script object to output data dict.
            script_4_stop = script_no.index(5)  # static tests have 4 scripts, so we add script 4 as well
            Script_4 = Script(time_vec[script_3_stop:script_4_stop], temperature[script_3_stop:script_4_stop],
                              voltage[script_3_stop:script_4_stop],
                              current[script_3_stop:script_4_stop], chgAh[script_3_stop:script_4_stop],
                              disAh[script_3_stop:script_4_stop])

            data['OCVData_' + temp] = [Script_1, Script_2, Script_3, Script_4]
            data['OCVData_full' + temp] = [time_vec, current, voltage, chgAh, disAh, temperature, script_no]
        else:
            data['DYNData_' + temp] = [Script_1, Script_2, Script_3]
            data['DYNData_full' + temp] = [time_vec, current, voltage, chgAh, disAh, temperature, script_no]

print('STEP 5: Saving the data dictionary into a .mat file: ' + output_data_filename)
scipy.io.savemat(output_data_filename, data)

print('Done!')
