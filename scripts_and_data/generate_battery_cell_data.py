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
# Todo static test fix because its hardcoded to work with Ts = 1e-2
Ts = 1e-2                                    # SP execution rate (Should be maximum 1/SIMULATION_SPEED_UP)
Ts_cell = Ts/5                               # Battery cells should run faster than the scripts
SIMULATION_SPEED_UP = 100                    # 1 if not sped up
capture_duration = 50e4/SIMULATION_SPEED_UP  # default for HIL (30*60*60) seconds, for VHIL full no hyst and no RC 23e4
M0Param = [0.0031315, 0.0023535, 0.0011502]
MParam = [0.039929, 0.020018, 0.020545]
R1 = 4e-3                                    # 0.64769e-3 default but in Coursera quizzes more like 8 mili-ohm
RC1 = 60/SIMULATION_SPEED_UP                 # RC is the time constant so it must be scaled (60 seconds seem right)
GParam = [67.207, 92.645, 67.840]            # Todo Should Gamma parameter depend on SIMULATION_SPEED_UP in some way?
numpoles = 1
doHyst = 0
test_temperatures = ['5', '25', '45']        # must exist in the model as well
# Numeric scale was configured to 1e2 instead of 1e6 (SOC calculation)

capture_rate = SIMULATION_SPEED_UP
output_data_filename = 'Typhoon_captured_data.mat'
model_name = "Battery_parametrization_model.tse"  # "Battery_parametrization_model.tse"
current_profile_filename = 'current_profiles.pickle'
flag_show = False  # if True, certain graphs used for debugging will be shown
vhil_device = True

# script directory
# Path to model file and to compiled model file
FILE_DIR_PATH = Path(__file__).parent
model_path = str(FILE_DIR_PATH / model_name)
compiled_model_path = model.get_compiled_model_file(model_path)

if numpoles == 0:
    R1 = 1e-5                # disabling RC poles
if doHyst == 0:
    M0Param = [1e-5, 1e-5, 1e-5]  # disabling hysteresis
    MParam = [1e-5, 1e-5, 1e-5]   # disabling hysteresis

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

OCV = [[ 2.81376455, 2.934249, 3.01131295, 3.0707406, 3.12232055, 3.1679386999999997, 3.20864785, 3.2450991, 3.27819715, 3.30856405, 3.3362725999999996, 3.3618633499999997, 3.38570235, 3.40775795, 3.42830425, 3.4476403500000004, 3.4655972, 3.48235125, 3.4982734000000004, 3.5123286499999997, 3.52428065, 3.5322503000000003, 3.536190725, 3.538614355, 3.5410027349999997, 3.543410505, 3.54570155, 3.548176695, 3.55074616, 3.55332738, 3.5560948150000002, 3.5590690300000003, 3.5622155269999998, 3.5656399850000002, 3.569116135, 3.5732180099999997, 3.57727722, 3.5820934199999996, 3.58702369, 3.5923769, 3.5981162299999996, 3.60387865, 3.6098619999999997, 3.61578425, 3.6219618000000002, 3.6279250050000003, 3.633843695, 3.6400532699999997, 3.646100845, 3.6525218749999997, 3.659006475, 3.6658932699999998, 3.6723717, 3.67845187, 3.6843657899999998, 3.690319095, 3.696598375, 3.70317101, 3.7103087649999997, 3.71751191, 3.72478503, 3.7312556949999998, 3.737126875, 3.7421215, 3.746753425, 3.7512470799999997, 3.7557191, 3.759948005, 3.7641692570000003, 3.7683243400000004, 3.77226982, 3.776130865, 3.77996065, 3.7834905, 3.78724775, 3.79068595, 3.79405435, 3.79739885, 3.8008169, 3.8037784, 3.80672915, 3.80976375, 3.81294495, 3.8155956, 3.81841755, 3.8211345, 3.82375915, 3.826463, 3.8289732, 3.83148675, 3.8339139, 3.8364958500000004, 3.8389766, 3.84130125, 3.8436186, 3.84593155, 3.8484112, 3.8508698999999997, 3.853017, 3.8553768, 3.8575925, 3.8597776, 3.8618757, 3.8639596, 3.8662289, 3.8684966, 3.8704239, 3.8727584, 3.8746679, 3.8767636000000003, 3.8789283, 3.88108645, 3.88336145, 3.88554035, 3.8876272999999997, 3.88979185, 3.8919731499999997, 3.89423455, 3.89660415, 3.89878675, 3.9010609, 3.9034554999999997, 3.9058273, 3.9083077999999998, 3.9107870499999997, 3.9134431, 3.91605185, 3.9188395000000003, 3.92173155, 3.9245562, 3.9278719, 3.93123635, 3.93501795, 3.9392118, 3.9434215, 3.9476625, 3.9522793, 3.956863, 3.9614228, 3.9661213500000003, 3.97056845, 3.97496085, 3.97923145, 3.98304985, 3.9867527999999997, 3.990355, 3.9938599000000004, 3.997107, 4.0001812, 4.0032136000000005, 4.0059932, 4.0087594499999994, 4.011432500000001, 4.0139152, 4.0163138, 4.0185217, 4.02070895, 4.022900900000001, 4.0249913, 4.0268188, 4.02872455, 4.03070295, 4.0323274, 4.0341031, 4.03572435, 4.0374238, 4.03904715, 4.0406617, 4.0421721999999995, 4.04385255, 4.04555215, 4.0469809, 4.048781399999999, 4.05039585, 4.052104, 4.05390415, 4.0557043, 4.057513849999999, 4.0593374, 4.0614338, 4.0635322, 4.0657283, 4.0680112, 4.0704244, 4.07274935, 4.075507399999999, 4.0782211, 4.0812930000000005, 4.08431865, 4.08759685, 4.0910807, 4.0947777499999995, 4.0989474, 4.10329865, 4.10799, 4.1129271, 4.1185746, 4.1244995, 4.1311105999999995, 4.138731625, 4.14890576], 
[2.8100227500000003, 2.916445, 2.99576475, 3.057303, 3.11000275, 3.1564935, 3.19803925, 3.2350955, 3.26858575, 3.2992202500000003, 3.3273629999999996, 3.35331675, 3.37731175, 3.39958975, 3.4203212499999998, 3.4398017500000004, 3.457986, 3.47495625, 3.4909670000000004, 3.50564325, 3.51860325, 3.5288515, 3.535353625, 3.539071775, 3.5418136749999998, 3.544252525, 3.54650775, 3.5488834750000002, 3.5513308, 3.5538369, 3.556474075, 3.55934515, 3.5622776349999996, 3.565399925, 3.5687806749999997, 3.57249005, 3.5763861, 3.5808671, 3.5855184500000004, 3.5906845, 3.5961811499999996, 3.60179325, 3.60771, 3.6137212499999998, 3.619809, 3.626025025, 3.632018475, 3.63826635, 3.644504225, 3.6510093749999997, 3.657832375, 3.66466635, 3.6710585, 3.67705935, 3.6830289499999997, 3.689195475, 3.6957918750000003, 3.70265505, 3.709943825, 3.71715955, 3.72392515, 3.729878475, 3.735234375, 3.7402075, 3.744967125, 3.7498354, 3.7545954999999998, 3.759340025, 3.764046285, 3.7688217, 3.7733491, 3.777854325, 3.78220325, 3.7862525, 3.7902387500000003, 3.79382975, 3.79747175, 3.80099425, 3.8044845, 3.807692, 3.81084575, 3.81401875, 3.81712475, 3.819978, 3.82288775, 3.8256725, 3.82839575, 3.831115, 3.833666, 3.83623375, 3.8387695, 3.8412792500000004, 3.843683, 3.84610625, 3.848493, 3.85085775, 3.853256, 3.8555495, 3.857885, 3.860084, 3.8623624999999997, 3.864488, 3.8665785, 3.868598, 3.8707445, 3.872883, 3.8749195, 3.876992, 3.8789395, 3.881018, 3.8830415, 3.88503225, 3.88720725, 3.88930175, 3.8913365, 3.89335925, 3.8954657499999996, 3.89757275, 3.89982075, 3.90193375, 3.9041045000000003, 3.9064775, 3.9087365, 3.911139, 3.91353525, 3.9160155, 3.91865925, 3.9213975000000003, 3.92425775, 3.927181, 3.9305595, 3.93418175, 3.93828975, 3.942859, 3.9479075, 3.9531125, 3.9585965, 3.9639149999999996, 3.968714, 3.97340675, 3.9776422499999997, 3.98160425, 3.98535725, 3.98884925, 3.992164, 3.995375, 3.9984995000000003, 4.001535, 4.004506, 4.007268, 4.0099659999999995, 4.01259725, 4.015162500000001, 4.017576, 4.019969, 4.0222085, 4.02434475, 4.026504500000001, 4.0285565, 4.030494, 4.03242275, 4.03431475, 4.036036999999999, 4.0377155, 4.03942175, 4.041119, 4.04283575, 4.0445085, 4.046061, 4.047662750000001, 4.04936075, 4.050904500000001, 4.052707, 4.05437925, 4.05612, 4.05792075, 4.0597215, 4.06156925, 4.063487, 4.065569, 4.067661, 4.0698415, 4.072056, 4.074522, 4.07694675, 4.079536999999999, 4.0823055, 4.085265000000001, 4.08839325, 4.0915842499999995, 4.0950035, 4.09868875, 4.102737, 4.106893250000001, 4.11155, 4.1162355, 4.1216729999999995, 4.1272975, 4.133553, 4.140458125, 4.149328799999999], 
[2.80628095, 2.898641, 2.98021655, 3.0438654, 3.09768495, 3.1450483, 3.18743065, 3.2250919, 3.2589743500000004, 3.2898764500000004, 3.3184533999999997, 3.34477015, 3.36892115, 3.39142155, 3.41233825, 3.43196315, 3.4503747999999996, 3.46756125, 3.4836606000000003, 3.4989578499999996, 3.5129258500000002, 3.5254527, 3.534516525, 3.539529195, 3.542624615, 3.545094545, 3.54731395, 3.549590255, 3.55191544, 3.55434642, 3.556853335, 3.55962127, 3.562339743, 3.565159865, 3.568445215, 3.57176209, 3.57549498, 3.5796407799999996, 3.58401321, 3.5889921, 3.5942460699999996, 3.59970785, 3.605558, 3.6116582499999996, 3.6176562, 3.624125045, 3.630193255, 3.63647943, 3.642907605, 3.6494968749999996, 3.656658275, 3.66343943, 3.6697452999999998, 3.67566683, 3.6816921099999997, 3.688071855, 3.694985375, 3.70213909, 3.709578885, 3.71680719, 3.72306527, 3.728501255, 3.7333418750000003, 3.7382934999999997, 3.743180825, 3.74842372, 3.7534718999999996, 3.758732045, 3.7639233130000003, 3.7693190600000004, 3.77442838, 3.779577785, 3.78444585, 3.7890145, 3.79322975, 3.7969735499999997, 3.80088915, 3.80458965, 3.8081521, 3.8116056, 3.81496235, 3.81827375, 3.8213045500000002, 3.8243603999999998, 3.82735795, 3.8302104999999997, 3.83303235, 3.8357669999999997, 3.8383588, 3.84098075, 3.8436251, 3.8460626500000004, 3.8483894000000003, 3.8509112500000002, 3.8533674, 3.85578395, 3.8581008, 3.8602290999999997, 3.862753, 3.8647912, 3.8671325, 3.8691984, 3.8712813, 3.8732364, 3.8752600999999998, 3.8772694, 3.8794151, 3.8812256, 3.8832111, 3.8852724000000003, 3.8871547, 3.88897805, 3.89105305, 3.8930631499999997, 3.8950457, 3.89692665, 3.8989583499999996, 3.90091095, 3.90303735, 3.90508075, 3.9071481, 3.9094995, 3.9116457000000002, 3.9139702, 3.91628345, 3.9185879, 3.92126665, 3.9239555, 3.92678395, 3.9298058, 3.9332471, 3.93712715, 3.9415615500000003, 3.9465062, 3.9523935, 3.9585624999999998, 3.9649137, 3.970967, 3.9760052, 3.9806921500000003, 3.98471605, 3.98824765, 3.99148305, 3.99464865, 3.9975752, 4.000395, 4.0031391, 4.005963, 4.0088308, 4.0113224, 4.0139388, 4.016435049999999, 4.018892500000001, 4.0212368, 4.0236241999999995, 4.0258953, 4.02798055, 4.0301081000000005, 4.032121699999999, 4.0341692, 4.03612095, 4.03792655, 4.0397466, 4.0413279, 4.04311915, 4.0448142, 4.04662435, 4.0483553, 4.0499498, 4.05147295, 4.05316935, 4.0548281, 4.0566325999999995, 4.05836265, 4.060136, 4.06193735, 4.0637387, 4.06562465, 4.0676366, 4.069704199999999, 4.0717898, 4.0739547, 4.0761008, 4.0786196, 4.08114415, 4.083566599999999, 4.0863899, 4.089237000000001, 4.09246785, 4.09557165, 4.0989263, 4.1025997499999995, 4.1065266, 4.11048785, 4.11511, 4.1195439, 4.1247714, 4.1300955, 4.1359954, 4.1421846250000005, 4.1497518399999995]]

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

    print('Done!')


# Default parameters from the Boulder Colorado data and octave script:
OCV_default = [[2.81376455, 2.934249, 3.01131295, 3.0707406, 3.12232055, 3.1679386999999997, 3.20864785, 3.2450991, 3.27819715, 3.30856405, 3.3362725999999996, 3.3618633499999997, 3.38570235, 3.40775795, 3.42830425, 3.4476403500000004, 3.4655972, 3.48235125, 3.4982734000000004, 3.5123286499999997, 3.52428065, 3.5322503000000003, 3.536190725, 3.538614355, 3.5410027349999997, 3.543410505, 3.54570155, 3.548176695, 3.55074616, 3.55332738, 3.5560948150000002, 3.5590690300000003, 3.5622155269999998, 3.5656399850000002, 3.569116135, 3.5732180099999997, 3.57727722, 3.5820934199999996, 3.58702369, 3.5923769, 3.5981162299999996, 3.60387865, 3.6098619999999997, 3.61578425, 3.6219618000000002, 3.6279250050000003, 3.633843695, 3.6400532699999997, 3.646100845, 3.6525218749999997, 3.659006475, 3.6658932699999998, 3.6723717, 3.67845187, 3.6843657899999998, 3.690319095, 3.696598375, 3.70317101, 3.7103087649999997, 3.71751191, 3.72478503, 3.7312556949999998, 3.737126875, 3.7421215, 3.746753425, 3.7512470799999997, 3.7557191, 3.759948005, 3.7641692570000003, 3.7683243400000004, 3.77226982, 3.776130865, 3.77996065, 3.7834905, 3.78724775, 3.79068595, 3.79405435, 3.79739885, 3.8008169, 3.8037784, 3.80672915, 3.80976375, 3.81294495, 3.8155956, 3.81841755, 3.8211345, 3.82375915, 3.826463, 3.8289732, 3.83148675, 3.8339139, 3.8364958500000004, 3.8389766, 3.84130125, 3.8436186, 3.84593155, 3.8484112, 3.8508698999999997, 3.853017, 3.8553768, 3.8575925, 3.8597776, 3.8618757, 3.8639596, 3.8662289, 3.8684966, 3.8704239, 3.8727584, 3.8746679, 3.8767636000000003, 3.8789283, 3.88108645, 3.88336145, 3.88554035, 3.8876272999999997, 3.88979185, 3.8919731499999997, 3.89423455, 3.89660415, 3.89878675, 3.9010609, 3.9034554999999997, 3.9058273, 3.9083077999999998, 3.9107870499999997, 3.9134431, 3.91605185, 3.9188395000000003, 3.92173155, 3.9245562, 3.9278719, 3.93123635, 3.93501795, 3.9392118, 3.9434215, 3.9476625, 3.9522793, 3.956863, 3.9614228, 3.9661213500000003, 3.97056845, 3.97496085, 3.97923145, 3.98304985, 3.9867527999999997, 3.990355, 3.9938599000000004, 3.997107, 4.0001812, 4.0032136000000005, 4.0059932, 4.0087594499999994, 4.011432500000001, 4.0139152, 4.0163138, 4.0185217, 4.02070895, 4.022900900000001, 4.0249913, 4.0268188, 4.02872455, 4.03070295, 4.0323274, 4.0341031, 4.03572435, 4.0374238, 4.03904715, 4.0406617, 4.0421721999999995, 4.04385255, 4.04555215, 4.0469809, 4.048781399999999, 4.05039585, 4.052104, 4.05390415, 4.0557043, 4.057513849999999, 4.0593374, 4.0614338, 4.0635322, 4.0657283, 4.0680112, 4.0704244, 4.07274935, 4.075507399999999, 4.0782211, 4.0812930000000005, 4.08431865, 4.08759685, 4.0910807, 4.0947777499999995, 4.0989474, 4.10329865, 4.10799, 4.1129271, 4.1185746, 4.1244995, 4.1311105999999995, 4.138731625, 4.14890576], [2.8100227500000003, 2.916445, 2.99576475, 3.057303, 3.11000275, 3.1564935, 3.19803925, 3.2350955, 3.26858575, 3.2992202500000003, 3.3273629999999996, 3.35331675, 3.37731175, 3.39958975, 3.4203212499999998, 3.4398017500000004, 3.457986, 3.47495625, 3.4909670000000004, 3.50564325, 3.51860325, 3.5288515, 3.535353625, 3.539071775, 3.5418136749999998, 3.544252525, 3.54650775, 3.5488834750000002, 3.5513308, 3.5538369, 3.556474075, 3.55934515, 3.5622776349999996, 3.565399925, 3.5687806749999997, 3.57249005, 3.5763861, 3.5808671, 3.5855184500000004, 3.5906845, 3.5961811499999996, 3.60179325, 3.60771, 3.6137212499999998, 3.619809, 3.626025025, 3.632018475, 3.63826635, 3.644504225, 3.6510093749999997, 3.657832375, 3.66466635, 3.6710585, 3.67705935, 3.6830289499999997, 3.689195475, 3.6957918750000003, 3.70265505, 3.709943825, 3.71715955, 3.72392515, 3.729878475, 3.735234375, 3.7402075, 3.744967125, 3.7498354, 3.7545954999999998, 3.759340025, 3.764046285, 3.7688217, 3.7733491, 3.777854325, 3.78220325, 3.7862525, 3.7902387500000003, 3.79382975, 3.79747175, 3.80099425, 3.8044845, 3.807692, 3.81084575, 3.81401875, 3.81712475, 3.819978, 3.82288775, 3.8256725, 3.82839575, 3.831115, 3.833666, 3.83623375, 3.8387695, 3.8412792500000004, 3.843683, 3.84610625, 3.848493, 3.85085775, 3.853256, 3.8555495, 3.857885, 3.860084, 3.8623624999999997, 3.864488, 3.8665785, 3.868598, 3.8707445, 3.872883, 3.8749195, 3.876992, 3.8789395, 3.881018, 3.8830415, 3.88503225, 3.88720725, 3.88930175, 3.8913365, 3.89335925, 3.8954657499999996, 3.89757275, 3.89982075, 3.90193375, 3.9041045000000003, 3.9064775, 3.9087365, 3.911139, 3.91353525, 3.9160155, 3.91865925, 3.9213975000000003, 3.92425775, 3.927181, 3.9305595, 3.93418175, 3.93828975, 3.942859, 3.9479075, 3.9531125, 3.9585965, 3.9639149999999996, 3.968714, 3.97340675, 3.9776422499999997, 3.98160425, 3.98535725, 3.98884925, 3.992164, 3.995375, 3.9984995000000003, 4.001535, 4.004506, 4.007268, 4.0099659999999995, 4.01259725, 4.015162500000001, 4.017576, 4.019969, 4.0222085, 4.02434475, 4.026504500000001, 4.0285565, 4.030494, 4.03242275, 4.03431475, 4.036036999999999, 4.0377155, 4.03942175, 4.041119, 4.04283575, 4.0445085, 4.046061, 4.047662750000001, 4.04936075, 4.050904500000001, 4.052707, 4.05437925, 4.05612, 4.05792075, 4.0597215, 4.06156925, 4.063487, 4.065569, 4.067661, 4.0698415, 4.072056, 4.074522, 4.07694675, 4.079536999999999, 4.0823055, 4.085265000000001, 4.08839325, 4.0915842499999995, 4.0950035, 4.09868875, 4.102737, 4.106893250000001, 4.11155, 4.1162355, 4.1216729999999995, 4.1272975, 4.133553, 4.140458125, 4.149328799999999], [2.80628095, 2.898641, 2.98021655, 3.0438654, 3.09768495, 3.1450483, 3.18743065, 3.2250919, 3.2589743500000004, 3.2898764500000004, 3.3184533999999997, 3.34477015, 3.36892115, 3.39142155, 3.41233825, 3.43196315, 3.4503747999999996, 3.46756125, 3.4836606000000003, 3.4989578499999996, 3.5129258500000002, 3.5254527, 3.534516525, 3.539529195, 3.542624615, 3.545094545, 3.54731395, 3.549590255, 3.55191544, 3.55434642, 3.556853335, 3.55962127, 3.562339743, 3.565159865, 3.568445215, 3.57176209, 3.57549498, 3.5796407799999996, 3.58401321, 3.5889921, 3.5942460699999996, 3.59970785, 3.605558, 3.6116582499999996, 3.6176562, 3.624125045, 3.630193255, 3.63647943, 3.642907605, 3.6494968749999996, 3.656658275, 3.66343943, 3.6697452999999998, 3.67566683, 3.6816921099999997, 3.688071855, 3.694985375, 3.70213909, 3.709578885, 3.71680719, 3.72306527, 3.728501255, 3.7333418750000003, 3.7382934999999997, 3.743180825, 3.74842372, 3.7534718999999996, 3.758732045, 3.7639233130000003, 3.7693190600000004, 3.77442838, 3.779577785, 3.78444585, 3.7890145, 3.79322975, 3.7969735499999997, 3.80088915, 3.80458965, 3.8081521, 3.8116056, 3.81496235, 3.81827375, 3.8213045500000002, 3.8243603999999998, 3.82735795, 3.8302104999999997, 3.83303235, 3.8357669999999997, 3.8383588, 3.84098075, 3.8436251, 3.8460626500000004, 3.8483894000000003, 3.8509112500000002, 3.8533674, 3.85578395, 3.8581008, 3.8602290999999997, 3.862753, 3.8647912, 3.8671325, 3.8691984, 3.8712813, 3.8732364, 3.8752600999999998, 3.8772694, 3.8794151, 3.8812256, 3.8832111, 3.8852724000000003, 3.8871547, 3.88897805, 3.89105305, 3.8930631499999997, 3.8950457, 3.89692665, 3.8989583499999996, 3.90091095, 3.90303735, 3.90508075, 3.9071481, 3.9094995, 3.9116457000000002, 3.9139702, 3.91628345, 3.9185879, 3.92126665, 3.9239555, 3.92678395, 3.9298058, 3.9332471, 3.93712715, 3.9415615500000003, 3.9465062, 3.9523935, 3.9585624999999998, 3.9649137, 3.970967, 3.9760052, 3.9806921500000003, 3.98471605, 3.98824765, 3.99148305, 3.99464865, 3.9975752, 4.000395, 4.0031391, 4.005963, 4.0088308, 4.0113224, 4.0139388, 4.016435049999999, 4.018892500000001, 4.0212368, 4.0236241999999995, 4.0258953, 4.02798055, 4.0301081000000005, 4.032121699999999, 4.0341692, 4.03612095, 4.03792655, 4.0397466, 4.0413279, 4.04311915, 4.0448142, 4.04662435, 4.0483553, 4.0499498, 4.05147295, 4.05316935, 4.0548281, 4.0566325999999995, 4.05836265, 4.060136, 4.06193735, 4.0637387, 4.06562465, 4.0676366, 4.069704199999999, 4.0717898, 4.0739547, 4.0761008, 4.0786196, 4.08114415, 4.083566599999999, 4.0863899, 4.089237000000001, 4.09246785, 4.09557165, 4.0989263, 4.1025997499999995, 4.1065266, 4.11048785, 4.11511, 4.1195439, 4.1247714, 4.1300955, 4.1359954, 4.1421846250000005, 4.1497518399999995]]
SOC_default = [0.00000, 0.00500, 0.01000, 0.01500, 0.02000, 0.02500, 0.03000, 0.03500, 0.04000, 0.04500, 0.05000, 0.05500, 0.06000, 0.06500, 0.07000, 0.07500, 0.08000, 0.08500, 0.09000, 0.09500, 0.10000, 0.10500, 0.11000, 0.11500, 0.12000, 0.12500, 0.13000, 0.13500, 0.14000, 0.14500, 0.15000, 0.15500, 0.16000, 0.16500, 0.17000, 0.17500, 0.18000, 0.18500, 0.19000, 0.19500, 0.20000, 0.20500, 0.21000, 0.21500, 0.22000, 0.22500, 0.23000, 0.23500, 0.24000, 0.24500, 0.25000, 0.25500, 0.26000, 0.26500, 0.27000, 0.27500, 0.28000, 0.28500, 0.29000, 0.29500, 0.30000, 0.30500, 0.31000, 0.31500, 0.32000, 0.32500, 0.33000, 0.33500, 0.34000, 0.34500, 0.35000, 0.35500, 0.36000, 0.36500, 0.37000, 0.37500, 0.38000, 0.38500, 0.39000, 0.39500, 0.40000, 0.40500, 0.41000, 0.41500, 0.42000, 0.42500, 0.43000, 0.43500, 0.44000, 0.44500, 0.45000, 0.45500, 0.46000, 0.46500, 0.47000, 0.47500, 0.48000, 0.48500, 0.49000, 0.49500, 0.50000, 0.50500, 0.51000, 0.51500, 0.52000, 0.52500, 0.53000, 0.53500, 0.54000, 0.54500, 0.55000, 0.55500, 0.56000, 0.56500, 0.57000, 0.57500, 0.58000, 0.58500, 0.59000, 0.59500, 0.60000, 0.60500, 0.61000, 0.61500, 0.62000, 0.62500, 0.63000, 0.63500, 0.64000, 0.64500, 0.65000, 0.65500, 0.66000, 0.66500, 0.67000, 0.67500, 0.68000, 0.68500, 0.69000, 0.69500, 0.70000, 0.70500, 0.71000, 0.71500, 0.72000, 0.72500, 0.73000, 0.73500, 0.74000, 0.74500, 0.75000, 0.75500, 0.76000, 0.76500, 0.77000, 0.77500, 0.78000, 0.78500, 0.79000, 0.79500, 0.80000, 0.80500, 0.81000, 0.81500, 0.82000, 0.82500, 0.83000, 0.83500, 0.84000, 0.84500, 0.85000, 0.85500, 0.86000, 0.86500, 0.87000, 0.87500, 0.88000, 0.88500, 0.89000, 0.89500, 0.90000, 0.90500, 0.91000, 0.91500, 0.92000, 0.92500, 0.93000, 0.93500, 0.94000, 0.94500, 0.95000, 0.95500, 0.96000, 0.96500, 0.97000, 0.97500, 0.98000, 0.98500, 0.99000, 0.99500, 1.00000]
R0Param = [4.6198e-3, 1.7810e-3, 1.1351e-3]
etaParam = [0.98174, 0.99102, 0.98965]
QParam = [14.592, 14.532, 14.444]
Rparam = [R1] * 3
RCparam = [RC1*SIMULATION_SPEED_UP] * 3
etaParam_static = [0.98174, 0.99102, 0.98965]
QParam_static = [14.592, 14.532, 14.444]

OCV_Typhoon = [2.7959764, 2.92032004, 3.00517251, 3.06865418, 3.12190095, 3.16862305, 3.21030271, 3.24750352, 3.28106613, 3.31171404, 3.33985442, 3.36579207, 3.38976362, 3.41200262, 3.43268866, 3.4520849 , 3.47021674, 3.48713713, 3.5030564 , 3.5177525 , 3.53085508, 3.54154451, 3.54882255, 3.55310383, 3.55592105, 3.55824688, 3.56035692, 3.56249258, 3.56471424, 3.56699753, 3.56938591, 3.57196933, 3.57467297, 3.57751755, 3.5805914 , 3.58396194, 3.58757009, 3.5915979 , 3.59595852, 3.60068474, 3.60581254, 3.61114488, 3.61669499, 3.62243072, 3.62824953, 3.63417081, 3.64003704, 3.64592675, 3.65193432, 3.65806845, 3.66448954, 3.6710776 , 3.67746585, 3.68344277, 3.68919648, 3.69501999, 3.70113753, 3.70760844, 3.71440755, 3.72142264, 3.72821616, 3.73442628, 3.73991974, 3.7449058 , 3.74957712, 3.75415275, 3.7587602 , 3.76329252, 3.76780196, 3.77230849, 3.77678371, 3.78108384, 3.78532269, 3.78936662, 3.79318191, 3.79684579, 3.80023761, 3.80363645, 3.80693904, 3.81015106, 3.81313712, 3.81608628, 3.81903616, 3.82188288, 3.82454075, 3.82722064, 3.82978901, 3.83230635, 3.83479374, 3.83714358, 3.839503  , 3.84183166, 3.84412484, 3.84632838, 3.84854527, 3.85072746, 3.85289324, 3.85508048, 3.85717544, 3.85930021, 3.86130316, 3.86326712, 3.86519041, 3.86704442, 3.8690165 , 3.87098652, 3.87285885, 3.87476108, 3.87654505, 3.87844917, 3.88030633, 3.88212964, 3.88412826, 3.88605734, 3.88792571, 3.88978024, 3.89171459, 3.89365292, 3.89572678, 3.89767618, 3.89967608, 3.90187288, 3.90396751, 3.9061962 , 3.90842413, 3.91073275, 3.91320224, 3.91576897, 3.91845651, 3.92120935, 3.92440474, 3.92785089, 3.93177549, 3.93616255, 3.94102865, 3.94606155, 3.95136997, 3.95652616, 3.96117301, 3.96570124, 3.96978205, 3.9735838 , 3.97717448, 3.98050546, 3.98365677, 3.98670239, 3.9896607 , 3.99253031, 3.99533452, 3.99793305, 4.00046426, 4.00292843, 4.00532679, 4.00757501, 4.00980004, 4.01187421, 4.01384386, 4.01583448, 4.01771955, 4.01949064, 4.02125076, 4.02297465, 4.02453118, 4.02604126, 4.02757803, 4.0291067 , 4.03065397, 4.03215879, 4.03354394, 4.03497594, 4.03650394, 4.03788079, 4.03951079, 4.04101595, 4.04258673, 4.04421792, 4.04584982, 4.04752798, 4.04927647, 4.05118789, 4.0531111 , 4.05512204, 4.05716773, 4.0594627 , 4.06171959, 4.06414004, 4.06673856, 4.06952867, 4.07248749, 4.07551026, 4.07876004, 4.08227605, 4.08615518, 4.09014336, 4.09463233, 4.09915541, 4.10444777, 4.11002323, 4.11671366, 4.12657983, 4.15168238]


def error_func(model_param, param_name):
    error = []
    for i in range(len(model_param)):
        if param_name == 'temps':
            error.append(round((int(test_temperatures[i])-model_param[i])/int(test_temperatures[i]), 2))
        elif param_name == 'R0Param':
            error.append(round((R0Param[i]-model_param[i])/R0Param[i], 2))
        elif param_name == 'etaParam':
            error.append(round((etaParam[i]-model_param[i])/etaParam[i], 2))
        elif param_name == 'QParam':
            error.append(round((QParam[i]-model_param[i])/QParam[i], 2))
        elif param_name == 'RParam':
            error.append(round((Rparam[i]-model_param[i])/Rparam[i], 2))
        elif param_name == 'RCParam':
            error.append(round((RCparam[i]-model_param[i])/RCparam[i], 2))
        elif param_name == 'etaParam_static':
            error.append(round((etaParam_static[i]-model_param[i])/etaParam_static[i], 2))
        elif param_name == 'QParam_static':
            error.append(round((QParam_static[i]-model_param[i])/QParam_static[i], 2))
        elif param_name == 'M0Param':
            error.append(round((M0Param[i]-model_param[i])/M0Param[i], 2))
        elif param_name == 'MParam':
            error.append(round((MParam[i]-model_param[i])/MParam[i], 2))
        elif param_name == 'GParam':
            error.append(round((GParam[i]-model_param[i])/GParam[i], 2))

    return error
