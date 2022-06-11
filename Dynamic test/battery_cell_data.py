import scipy.io

# more to come


class battery_model:
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


class OneTempDynData:  # DynamicData
    def __init__(self, MAT_data, temp=25):
        self.temp = temp
        self.script1 = DynScriptData_12(MAT_data['DYNData'][0][0][0])
        self.script2 = DynScriptData_12(MAT_data['DYNData'][0][0][1])
        self.script3 = DynScriptData_3(MAT_data['DYNData'][0][0][2])
        self.Z = []
        self.OCV = []
        self.Q = []
        self.eta = []


class DynScriptData_12:  # data per script
    def __init__(self, script):
        self.time = script[0][0][1][0]
        self.step = script[0][0][3][0]
        self.current = script[0][0][5][0]
        self.voltage = script[0][0][7][0]
        self.chgAh = script[0][0][9][0]
        self.disAh = script[0][0][11][0]

        # Raw values seem not to be needed

        # self.rawTime = script[0][0][0]
        # self.rawStep = script[0][0][2]
        # self.rawCurrent = script[0][0][4]
        # self.rawVoltage = script[0][0][6]
        # self.rawChgAh = script[0][0][8]
        # self.rawDisAh = script[0][0][10]

        # for index, i in enumerate(self.rawTime):
        #     self.rawTime[index] = self.rawTime[index][0]
        #     self.rawStep[index] = self.rawStep[index][0]
        #     self.rawCurrent[index] = self.rawCurrent[index][0]
        #     self.rawVoltage[index] = self.rawVoltage[index][0]
        #     self.rawChgAh[index] = self.rawChgAh[index][0]
        #     self.rawDisAh[index] = self.rawDisAh[index][0]


class DynScriptData_3:  # data per script
    def __init__(self, script):
        self.time = script[0][0][0][0]
        self.voltage = script[0][0][1][0]
        self.current = script[0][0][2][0]
        self.chgAh = script[0][0][3][0]
        self.disAh = script[0][0][4][0]
        self.step = script[0][0][5][0]


class OneTempStaticData:  # Static data
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
    def __init__(self, script):
        self.time = []  # initiate empty list
        self.step = []  # initiate empty list
        self.current = []  # initiate empty list
        self.voltage = []  # initiate empty list
        self.chgAh = []  # initiate empty list
        self.disAh = []  # initiate empty list

        for i in range(len(script[0][0][0])):
            self.time.append(script[0][0][0][i][0])  # array of list elements with one element -> array of elements
            self.step.append(script[0][0][1][i][0])  # array of list elements with one element -> array of elements
            self.current.append(script[0][0][2][i][0])  # array of list elements with one element -> array of elements
            self.voltage.append(script[0][0][3][i][0])  # array of list elements with one element -> array of elements
            self.chgAh.append(script[0][0][4][i][0])  # array of list elements with one element -> array of elements
            self.disAh.append(script[0][0][5][i][0])  # array of list elements with one element -> array of elements


P14_DYN_50_P45 = OneTempDynData(scipy.io.loadmat("P14_DYN_50_P45.mat"), 45)
P14_DYN_50_P25 = OneTempDynData(scipy.io.loadmat("P14_DYN_50_P25.mat"), 25)
P14_DYN_30_P05 = OneTempDynData(scipy.io.loadmat("P14_DYN_30_P05.mat"), 5)
P14_OCV_P45 = OneTempStaticData(scipy.io.loadmat("P14_OCV_P45.mat"), 45)
P14_OCV_P25 = OneTempStaticData(scipy.io.loadmat("P14_OCV_P25.mat"), 25)
P14_OCV_P05 = OneTempStaticData(scipy.io.loadmat("P14_OCV_P05.mat"), 5)
P14_model = battery_model()
