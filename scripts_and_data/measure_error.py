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
import generate_pickled_cell_model
import generate_battery_cell_data


# This script will compare the dynamic data voltage between the original model and the new model, where the new model
# has the parsed out data from the dynamic and static scripts.

data_origin = 'Typhoon_captured_data.pickle'  # 'Typhoon Hil software and hardware obtained data'
pickled_data = open(data_origin, "rb")
Typhoon_captured_data = pickle.load(pickled_data)

# Parameters:
M0Param = Typhoon_captured_data.M0Param
MParam = Typhoon_captured_data.MParam
GParam = Typhoon_captured_data.GParam

R1 = Typhoon_captured_data.RParam[0]
RC1 = Typhoon_captured_data.RCParam[0]

Q = Typhoon_captured_data.QParam_static
eta = Typhoon_captured_data.etaParam_static

R0 = Typhoon_captured_data.R0Param
OCV = Typhoon_captured_data.ocv_vector
for index, i in enumerate(OCV):  # Small reformatting from arrays to lists
    OCV[index] = list(OCV[index])
temps = Typhoon_captured_data.temps
test_temperatures = ['5', '25', '45']        # must exist in the model as well

Ts = 1e-2                                    # SP execution rate (Should be maximum 1/SIMULATION_SPEED_UP)
Ts_cell = Ts/5                               # Battery cells should run faster than the scripts
SIMULATION_SPEED_UP = 100                    # 1 if not sped up
capture_duration = 50e4/SIMULATION_SPEED_UP  # default for HIL (30*60*60) seconds, for VHIL full no hyst and no RC 23e4
RC1 = RC1/SIMULATION_SPEED_UP                 # RC is the time constant so it must be scaled (60 seconds seem right)
numpoles = generate_battery_cell_data.numpoles
doHyst = generate_battery_cell_data.doHyst

capture_rate = SIMULATION_SPEED_UP
output_data_filename = 'Typhoon_captured_data_validation.mat'
model_name = generate_battery_cell_data.model_name  # "Battery_parametrization_model.tse"
current_profile_filename = generate_battery_cell_data.current_profile_filename
flag_show = False  # if True, certain graphs used for debugging will be shown
vhil_device = generate_battery_cell_data.vhil_device

# script directory
# Path to model file and to compiled model file
FILE_DIR_PATH = Path(__file__).parent
model_path = str(FILE_DIR_PATH / model_name)
compiled_model_path = model.get_compiled_model_file(model_path)

model_init_code = f"""
R1 = {str(R1)}
C1 = {str(RC1/R1)}
M0 = {str(M0Param)}
M = {str(MParam)}
SIMULATION_SPEED_UP = {str(SIMULATION_SPEED_UP)}
init_dynamic = 100  # 90% initial state of charge for dynamic scripts so that the voltage doesnt go over Vmax
Ts = {str(Ts)}
Ts_cell = {str(Ts_cell)}
G = {str(GParam)}  # G param is wrong for default battery
do_dyn_tither = {str(doHyst)}  # dynamic tither makes no sense if there is no hysteresis

rest_time_for_temperature_equilibrium = 7200/SIMULATION_SPEED_UP  # 7200s or 2h
counter_cooldown_max = rest_time_for_temperature_equilibrium/Ts
counter_cooldown_max = counter_cooldown_max/SIMULATION_SPEED_UP

OCV = {OCV}

Vmin_5 = min(OCV[0])  # Volts
Vmin_25 = min(OCV[1]) # Volts
Vmin_45 = min(OCV[2]) # Volts
Vmax_5 = max(OCV[0])  # Volts
Vmax_25 = max(OCV[1]) # Volts
Vmax_45 = max(OCV[2]) # Volts

total_Q_original = [14.592, 14.532, 14.444]
total_Q = [i/SIMULATION_SPEED_UP for i in total_Q_original]

C = 14.532  # C rate of a battery is current that will empty the cell in one hour
DISCHG_RATE =  0.5 # 0.5 if its normal battery discharg speed
CHG_RATE = -DISCHG_RATE

# when the dynamic script executes, current profiles don't push the cell voltage
# over the intended Vmax
V_end_script_1 = (Vmax_5-Vmin_5)*0.01+Vmin_5  # ends dynamic script 1 when SOC is roughly at 10%
DYN_DIS_CHG = C/30
DYN_CHG = -C
print(V_end_script_1)
info(V_end_script_1)

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
    print('STEP 1: Initializing the model')
    # Run very fast simulations and thus to obtain the data quickly before the physical tests take place
    model.load(model_path)
    model_device = model.get_model_property_value("hil_device")
    model_config = model.get_model_property_value("hil_configuration_id")
    report.report_message("Virtual HIL device is used. Model is compiled for {} C{}.".format(model_device, model_config))
    model.set_model_init_code(model_init_code)
    model.save_as("Typhoon_captured_data_validation.tse")  # saving a model

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

            voltage[0] = voltage[1]  # this line of code is necessary because the voltage from probe starts with 0

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

    print('STEP 4b: Adding the data about hysteresis and numpoles')
    data['doHyst'] = doHyst
    data['numpoles'] = numpoles

    print('STEP 5: Saving the data dictionary into a .mat file: ' + output_data_filename)
    scipy.io.savemat(output_data_filename, data)

    print('Comparing the results of the validation and reference data')
    TYPHOON_REFERENCE = data.CellAllData(scipy.io.loadmat("Typhoon_captured_data" + ".mat"), [5, 25, 45], [5, 25, 45])
    TYPHOON_VALIDATION = data.CellAllData(scipy.io.loadmat("Typhoon_captured_data_validation" + ".mat"), [5, 25, 45], [5, 25, 45])
    for k in range(len(temps)):
        TYPHOON_REFERENCE_CURRENT = TYPHOON_REFERENCE.dynamic_data[k].script1.current
        TYPHOON_VALIDATION_CURRENT = TYPHOON_VALIDATION.dynamic_data[k].script1.current
        RMS_error = sum((TYPHOON_VALIDATION_CURRENT-TYPHOON_REFERENCE_CURRENT)**2)/len(TYPHOON_VALIDATION_CURRENT)
        print(f"{RMS_error=}")
        plt.plot(TYPHOON_REFERENCE_CURRENT)
        plt.plot(TYPHOON_VALIDATION_CURRENT)
    plt.show()
