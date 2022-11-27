import scipy.io
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import optimize
fminbnd = scipy.optimize.fminbound
from scipy import interpolate
import matplotlib.pyplot as plt
import generate_battery_cell_data as cell_data


def plot_func(x_axis_list, y_axis_list, names, flag_show: bool = False):
    """
    Plotting function template that makes it easier to plot multiple graphs with less code
    """
    if flag_show:
        for i in range(len(x_axis_list)):
            # Optional, but must be between subplot and show
            plt.title(names[i])
            plt.grid(True)
            plt.plot(x_axis_list[i], y_axis_list[i], linewidth=2.0)

        plt.show()  # Last line


def error_func(model_param, param_name):
    if param_name == 'OCV':
        error = sum((np.array(cell_data.OCV_default[1])-np.array(model_param))**2)  # RMS of entire OCV
    else:
        error = []
        for i in range(len(model_param)):
            if param_name == 'temps':
                error.append(round((int(cell_data.test_temperatures[i])-model_param[i])/int(cell_data.test_temperatures[i]), 2))
            elif param_name == 'R0Param':
                error.append(round((cell_data.R0Param[i]-model_param[i])/cell_data.R0Param[i], 2))
            elif param_name == 'etaParam':
                error.append(round((cell_data.etaParam[i]-model_param[i])/cell_data.etaParam[i], 2))
            elif param_name == 'QParam':
                error.append(round((cell_data.QParam[i]-model_param[i])/cell_data.QParam[i], 2))
            elif param_name == 'RParam':
                error.append(round((cell_data.Rparam[i]-model_param[i])/cell_data.Rparam[i], 2))
            elif param_name == 'RCParam':
                error.append(round((cell_data.RCparam[i]-model_param[i])/cell_data.RCparam[i], 2))
            elif param_name == 'etaParam_static':
                error.append(round((cell_data.etaParam_static[i]-model_param[i])/cell_data.etaParam_static[i], 2))
            elif param_name == 'QParam_static':
                error.append(round((cell_data.QParam_static[i]-model_param[i])/cell_data.QParam_static[i], 2))
            elif param_name == 'M0Param':
                error.append(round((cell_data.M0Param[i]-model_param[i])/cell_data.M0Param[i], 2))
            elif param_name == 'MParam':
                error.append(round((cell_data.MParam[i]-model_param[i])/cell_data.MParam[i], 2))
            elif param_name == 'GParam':
                error.append(round((cell_data.GParam[i]-model_param[i])/cell_data.GParam[i], 2))

    return error


class ESC_battery_model:
    """
    ESC battery model to be saved as a pickled file
    """

    def __init__(self):
        self.temps = []
        self.QParam = []
        self.etaParam = []
        self.QParam_static = []
        self.etaParam_static = []
        self.RCParam = []
        self.RParam = []
        self.soc_vector = []  # should be multidimensional arrays
        self.ocv_vector = []  # should be multidimensional arrays
        self.GParam = []  # model.GParam = [0] * len(data)
        self.M0Param = []  # model.M0Param = [0] * len(data)
        self.MParam = []  # model.MParam = [0] * len(data)
        self.R0Param = []  # model.R0Param = [0] * len(data)
        self.RCParam = []  # model.RCParam = [[0] * len(data)] * numpoles
        self.RParam = []  # model.RParam = [[0] * len(data)] * numpoles
        self.CParam = []  # model.RParam = [[0] * len(data)] * numpoles


class OneTempDynData:
    """
    Dynamic data for a single temperature value. Consists of 3 scripts from dynamic tests.
    """

    def __init__(self, MAT_data, temp=25):
        self.temp = temp
        self.script1 = DynScriptData_12(MAT_data['DYNData'][0][0][0])
        self.script2 = DynScriptData_12(MAT_data['DYNData'][0][0][1])
        self.script3 = DynScriptData_3(MAT_data['DYNData'][0][0][2])
        self.Z = []
        self.OCV = []
        self.Q = []
        self.eta = []


class DynScriptData_12:
    """
    Dynamic data for scripts 1 and 2
    """

    def __init__(self, script):
        self.time = script[0][0][1][0]
        self.step = script[0][0][3][0]
        self.current = script[0][0][5][0]
        self.voltage = script[0][0][7][0]
        self.chgAh = script[0][0][9][0]
        self.disAh = script[0][0][11][0]


class DynScriptData_3:
    """
    Dynamic data for script 3
    """

    def __init__(self, script):
        self.time = script[0][0][0][0]
        self.voltage = script[0][0][1][0]
        self.current = script[0][0][2][0]
        self.chgAh = script[0][0][3][0]
        self.disAh = script[0][0][4][0]
        self.step = script[0][0][5][0]


class OneTempStaticData:
    """
    Static data for a single temperature value. Consists of 4 scripts from static tests.
    """

    def __init__(self, MAT_data, temp=25):
        self.temp = temp
        self.script1 = StaticScriptData(MAT_data['OCVData'][0][0][0])
        self.script2 = StaticScriptData(MAT_data['OCVData'][0][0][1])
        self.script3 = StaticScriptData(MAT_data['OCVData'][0][0][2])
        self.script4 = StaticScriptData(MAT_data['OCVData'][0][0][3])
        self.Z = []
        self.OCV = []
        self.Q = []
        self.eta = []


class StaticScriptData:
    """
    Static data for all scripts
    """

    def __init__(self, script):
        self.time = []
        self.step = []
        self.current = []
        self.voltage = []
        self.chgAh = []
        self.disAh = []

        for i in range(len(script[0][0][0])):
            self.time.append(script[0][0][0][i][0])  # array of list elements with one element -> array of elements
            self.step.append(script[0][0][1][i][0])  # array of list elements with one element -> array of elements
            self.current.append(script[0][0][2][i][0])  # array of list elements with one element -> array of elements
            self.voltage.append(script[0][0][3][i][0])  # array of list elements with one element -> array of elements
            self.chgAh.append(script[0][0][4][i][0])  # array of list elements with one element -> array of elements
            self.disAh.append(script[0][0][5][i][0])  # array of list elements with one element -> array of elements


class CellAllData:
    """
    Data class of a battery cell that contains all the battery cell data.
    This type of class is used to store the data obtained from the generated cell data from the Typhoon software.
    """

    def __init__(self, MAT_data, temp_static=[], temp_dyn=[]):
        self.static_data = []
        self.dynamic_data = []
        self.doHyst = MAT_data['doHyst'][0][0]
        self.numpoles = MAT_data['numpoles'][0][0]

        try:
            for temp in temp_static:
                if temp == 6:
                    self.static_data.append(self.StaticData(MAT_data['OCVData_' + '05'], temp))
                else:
                    self.static_data.append(self.StaticData(MAT_data['OCVData_' + str(temp)], temp))
        except ValueError:
            print('ValueError for static data. Check the format of the provided data')

        try:
            for temp in temp_dyn:
                if temp == 6:
                    self.dynamic_data.append(self.DynData(MAT_data['DYNData_' + '05'], temp))
                else:
                    self.dynamic_data.append(self.DynData(MAT_data['DYNData_' + str(temp)], temp))
        except ValueError:
            print('ValueError for dynamic data. Check the format of the provided data')

    class DynData(OneTempDynData):
        """
        dynamic set of data for one temperature
        """

        def __init__(self, MAT_data, temp=25):
            self.temp = temp
            self.script1 = self.ScriptData(MAT_data[0][0])
            self.script2 = self.ScriptData(MAT_data[0][1])
            self.script3 = self.ScriptData(MAT_data[0][2])
            self.Z = []
            self.OCV = []
            self.Q = []
            self.eta = []

        class ScriptData:
            """
            Dynamic data for scripts 1, 2 and 3
            """

            # time, temp, voltage, current, chgAh, disAh
            def __init__(self, script):
                self.time = script[0][0][0][0]
                self.temp = script[0][0][1][0]
                self.voltage = script[0][0][2][0]
                self.current = script[0][0][3][0]
                self.chgAh = script[0][0][4][0]
                self.disAh = script[0][0][5][0]
                self.OCV_real = script[0][0][6][0]

    class StaticData(OneTempStaticData):
        """
        static set of data for one temperature
        """

        def __init__(self, MAT_data, temp=25):
            self.temp = temp
            self.script1 = self.ScriptData(MAT_data[0][0])
            self.script2 = self.ScriptData(MAT_data[0][1])
            self.script3 = self.ScriptData(MAT_data[0][2])
            self.script4 = self.ScriptData(MAT_data[0][3])
            self.Z = []
            self.OCV = []
            self.Q = []
            self.eta = []

        class ScriptData:
            """
            Static data for all scripts
            time, temp, voltage, current, chgAh, disAh
            """

            # time, temp, voltage, current, chgAh, disAh
            def __init__(self, script):
                self.time = script[0][0][0][0]
                self.temp = script[0][0][1][0]
                self.voltage = script[0][0][2][0]
                self.current = script[0][0][3][0]
                self.chgAh = script[0][0][4][0]
                self.disAh = script[0][0][5][0]


class Script:  # Format for the pickled cell data is this class per temperature, per script
    def __init__(self, time, temperature, voltage, current, chgAh, disAh, OCV=None):
        if OCV is None:
            OCV = voltage
        self.time = time
        self.temperature = temperature
        self.voltage = voltage
        self.current = current
        self.chgAh = chgAh
        self.disAh = disAh
        self.OCV = OCV
