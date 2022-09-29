import scipy.io
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import optimize
fminbnd = scipy.optimize.fminbound
from scipy import interpolate


def plot_func(x_axis_list, y_axis_list, names, flag_show: bool = False):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 11))
    for i in range(len(x_axis_list)):
        plt.subplot(221 + i)
        # Optional, but must be between subplot and show
        plt.title(names[i])
        plt.grid(True)
        plt.plot(x_axis_list[i], y_axis_list[i], linewidth=3.0)

    # Optional spacing options
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    if flag_show:
        plt.show()  # Last line


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

        try:
            for temp in temp_static:
                if temp == 5:
                    self.static_data.append(self.StaticData(MAT_data['OCVData_' + '05'], temp))
                else:
                    self.static_data.append(self.StaticData(MAT_data['OCVData_' + str(temp)], temp))

        except ValueError:
            print('ValueError for static data. Check the format of the provided data')

        try:
            for temp in temp_dyn:
                if temp == 5:
                    self.static_data.append(self.DynData(MAT_data['DYNData_' + '05'], temp))
                else:
                    self.static_data.append(self.DynData(MAT_data['DYNData_' + str(temp)], temp))

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
                self.time = script[0][0][0]
                self.voltage = script[0][2][0]
                self.current = script[0][3][0]
                self.chgAh = script[0][4][0]
                self.disAh = script[0][5][0]

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
                self.time = script[0][0][0]
                self.voltage = script[0][2][0]
                self.current = script[0][3][0]
                self.chgAh = script[0][4][0]
                self.disAh = script[0][5][0]
