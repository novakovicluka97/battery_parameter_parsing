import scipy.io


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
        self.GParam = []      # model.GParam = [0] * len(data)
        self.M0Param = []     # model.M0Param = [0] * len(data)
        self.MParam = []      # model.MParam = [0] * len(data)
        self.R0Param = []     # model.R0Param = [0] * len(data)
        self.RCParam = []     # model.RCParam = [[0] * len(data)] * numpoles
        self.RParam = []      # model.RParam = [[0] * len(data)] * numpoles


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
            self.time.append(script[0][0][0][i][0])     # array of list elements with one element -> array of elements
            self.step.append(script[0][0][1][i][0])     # array of list elements with one element -> array of elements
            self.current.append(script[0][0][2][i][0])  # array of list elements with one element -> array of elements
            self.voltage.append(script[0][0][3][i][0])  # array of list elements with one element -> array of elements
            self.chgAh.append(script[0][0][4][i][0])    # array of list elements with one element -> array of elements
            self.disAh.append(script[0][0][5][i][0])    # array of list elements with one element -> array of elements


# Initialize data
P14_DYN_50_P45 = OneTempDynData(scipy.io.loadmat("P14_DYN_50_P45.mat"), 45)
P14_DYN_50_P25 = OneTempDynData(scipy.io.loadmat("P14_DYN_50_P25.mat"), 25)
P14_DYN_30_P05 = OneTempDynData(scipy.io.loadmat("P14_DYN_30_P05.mat"), 5)
P14_OCV_P45 = OneTempStaticData(scipy.io.loadmat("P14_OCV_P45.mat"), 45)
P14_OCV_P25 = OneTempStaticData(scipy.io.loadmat("P14_OCV_P25.mat"), 25)
P14_OCV_P05 = OneTempStaticData(scipy.io.loadmat("P14_OCV_P05.mat"), 5)

# Initialize model
P14_model = ESC_battery_model()
