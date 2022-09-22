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


def processStatic(static_data, model):
    """
    Script that populates the model parameters based on static test data
    """
    SOC_vector = np.linspace(0, 1, 201)  # output SOC_vector points for this step

    # We need to find the index of static data which coresponds with default temperature of 25 degrees celsius
    # After that, other temperatures can be calculated
    ind25 = None
    for index, single_temp_data in enumerate(static_data):
        if single_temp_data.temp == 25:
            ind25 = index
    if not ind25:
        print("There is no default temperature of 25 degrees celsius!")
        raise Exception()

    totDisAh = static_data[ind25].script1.disAh[-1] + static_data[ind25].script2.disAh[-1] + \
               static_data[ind25].script3.disAh[-1] + static_data[ind25].script4.disAh[-1]
    totChgAh = static_data[ind25].script1.chgAh[-1] + static_data[ind25].script2.chgAh[-1] + \
               static_data[ind25].script3.chgAh[-1] + static_data[ind25].script4.chgAh[-1]
    eta25 = totDisAh / totChgAh
    # Q doesn't need to be calculated differently for temp = 25

    for k in range(len(static_data)):
        # First step: calculate eta
        # scripts 1 and 3 are at test temperature but scripts 2 and 4 are always at 25 degrees
        totDisAh = static_data[k].script1.disAh[-1] + static_data[k].script2.disAh[-1] + static_data[k].script3.disAh[-1] + static_data[k].script4.disAh[-1]
        totChgAh_at_X_temp = static_data[k].script1.chgAh[-1] + static_data[k].script3.chgAh[-1]
        totChgAh_at_25_temp = static_data[k].script2.chgAh[-1] + static_data[k].script4.chgAh[-1]
        eta = (totDisAh - totChgAh_at_25_temp*eta25) / totChgAh_at_X_temp

        # Second step: calculate Q
        Q = static_data[k].script1.disAh[-1] + static_data[k].script2.disAh[-1] - eta * static_data[k].script1.chgAh[-1] - eta * static_data[k].script2.chgAh[-1]

        # Third step: calculate OCV curve
        print(f"Calculating static tests for temperature: {static_data[k].temp}")
        indD  = np.where(np.array(static_data[k].script1.step) == 2)[0]  # slow discharge
        IR1Da = static_data[k].script1.voltage[indD[0] - 1] - static_data[k].script1.voltage[indD[0]]
        IR2Da = static_data[k].script1.voltage[indD[-1] + 1] - static_data[k].script1.voltage[indD[-1]]

        indC  = np.where(np.array(static_data[k].script3.step) == 2)[0]  # slow charge
        IR1Ca = static_data[k].script3.voltage[indC[0]] - static_data[k].script3.voltage[indC[0] - 1]
        IR2Ca = static_data[k].script3.voltage[indC[-1]] - static_data[k].script3.voltage[indC[-1] + 1]
        IR1D = min(IR1Da, 2*IR2Ca)
        IR2D = min(IR2Da, 2*IR1Ca)
        IR1C = min(IR1Ca, 2*IR2Da)
        IR2C = min(IR2Ca, 2*IR1Da)

        blend = np.array(range(len(indD)))/(len(indD)-1)
        IRblend = IR1D + (IR2D-IR1D)*blend
        disV = static_data[k].script1.voltage[indD[0]:(indD[-1] + 1)] + IRblend
        disZ = 1 - static_data[k].script1.disAh[indD[0]:(indD[-1] + 1)] / Q
        disZ = disZ + (1 - disZ[0])

        blend = np.array(range(len(indC)))/(len(indC)-1)
        IRblend = IR1C + (IR2C-IR1C)*blend
        chgV = static_data[k].script3.voltage[indC[0]:indC[-1] + 1] - IRblend
        chgZ = static_data[k].script3.chgAh[indC[0]:indC[-1] + 1] / Q
        chgZ = chgZ - chgZ[0]

        Voltage_SOC_curve = interpolate.interp1d(chgZ, chgV)
        Discharge_SOC_curve = interpolate.interp1d(disZ, disV)
        deltaV50 = Voltage_SOC_curve(0.5) - Discharge_SOC_curve(0.5)
        ind = np.where(chgZ < 0.5)[0]
        vChg = chgV[ind[0]:ind[-1]] - chgZ[ind[0]:ind[-1]]*deltaV50
        zChg = chgZ[ind[0]:ind[-1]]
        ind = np.where(disZ > 0.5)[0]
        vDis = np.flipud(disV[ind[0]:ind[-1]] + (1 - disZ[ind[0]:ind[-1]])*deltaV50)
        zDis = np.flipud(disZ[ind[0]:ind[-1]])
        np.append(zChg, zDis)
        FULL_SOC_curve = interpolate.interp1d(np.append(zChg, zDis), np.append(vChg, vDis), fill_value="extrapolate")
        rawocv = FULL_SOC_curve(SOC_vector)

        # Final step: populate model
        model.temps.append(static_data[k].temp)
        model.etaParam_static.append(eta)
        model.QParam_static.append(Q)
        model.soc_vector.append(SOC_vector)
        model.ocv_vector.append(rawocv)


def processDynamic(data, model, numpoles, doHyst):
    """
    Script that populates the model parameters based on dynamic test data
    """
    # First step: Compute Q and eta
    ind25 = None
    for index, single_temp_data in enumerate(data):
        if single_temp_data.temp == 25:
            ind25 = index
    if not ind25:
        print("There is no default temperature of 25 deg celsius!")
        raise Exception()

    totDisAh = data[ind25].script1.disAh[-1] + data[ind25].script2.disAh[-1] + data[ind25].script3.disAh[-1]
    totChgAh = data[ind25].script1.chgAh[-1] + data[ind25].script2.chgAh[-1] + data[ind25].script3.chgAh[-1]
    eta25 = totDisAh / totChgAh

    data[ind25].script1.chgAh[-1] = eta25 * data[ind25].script1.chgAh[-1]  # correct for the coefficients
    data[ind25].script2.chgAh[-1] = eta25 * data[ind25].script2.chgAh[-1]  # correct for the coefficients
    data[ind25].script3.chgAh[-1] = eta25 * data[ind25].script3.chgAh[-1]  # correct for the coefficients
    Q25 = data[ind25].script1.disAh[-1] + data[ind25].script2.disAh[-1] - data[ind25].script1.chgAh[-1] - data[ind25].script2.chgAh[-1]
    data[ind25].Q = Q25
    data[ind25].eta = eta25

    for k in range(len(data)):
        if data[k].temp != 25:
            data[k].script2.chgAh = data[k].script2.chgAh * eta25
            data[k].script3.chgAh = data[k].script3.chgAh * eta25
            eta = (data[k].script1.disAh[-1] + data[k].script2.disAh[-1] +
                   data[k].script3.disAh[-1] - data[k].script2.chgAh[-1] -
                   data[k].script3.chgAh[-1]) / data[k].script1.chgAh[-1]

            data[k].script1.chgAh = eta * data[k].script1.chgAh
            Q = data[k].script1.disAh[-1] + data[k].script2.disAh[-1] - data[k].script1.chgAh[-1] - data[k].script2.chgAh[-1]
            data[k].Q = Q
            data[k].eta = eta

        # Populate model with parameters from the dynamic data
        model.QParam.append(data[k].Q)
        model.etaParam.append(data[k].eta)
        if model.temps[k] != data[k].temp:
            print("Dynamic data temperatures are not matching the static data temperatures or their numbering order.")
            raise Exception()

    # Second step: Compute OCV for "discharge portion" of test, based on static test data
    for k in range(len(data)):
        corrected_current = [0]*len(data[k].script1.current)
        for index, current in enumerate(data[k].script1.current):
            if current < 0:  # if current is flowing into the battery and charging it, apply a coefficient
                corrected_current[index] = current * model.etaParam[k]
            else:
                corrected_current[index] = current
        data[k].Z = np.ones(np.size(corrected_current)) - np.cumsum(corrected_current)/(data[k].Q*3600)
        data[k].OCV = OCVfromSOCtemp(data[k].Z, data[k].temp, model)
        # plt.plot(data[k].script1.time, data[k].Z)

    # Third step: Use optimization algorythm to find parameters M, M0, G, RC, R and R0
    model.GParam = [0] * len(data)
    model.MParam = [0] * len(data)
    model.M0Param = [0] * len(data)
    model.RParam = [0] * len(data)
    model.R0Param = [0] * len(data)
    model.RCParam = [0] * len(data)
    for k in range(len(data)):
        global bestcost
        bestcost = np.inf
        print("Processing temperature: ", data[k].temp)
        if doHyst:
            model.GParam[k] = abs(fminbnd(optfn, 1, 250, args=(data, model, model.temps[k], doHyst), xtol=0.1, maxfun=40, disp=0))
        else:  # Todo check functionality and extend it if it doesnt work
            model.GParam.append(0)
            theGParam = 0
            optfn(theGParam, data, model, model.temps[k], doHyst)

        [_, model] = minfn(data, model, model.temps[k], doHyst)

    print("Dynamic model created!")
    return model


def OCVfromSOCtemp(soc, temp, model):
    """
    Extrapolates OCV vector from the soc vector but based on the previous static OCV/SOC test contained in the model.
    """
    index = np.where(np.array(model.temps) == temp)[0][0]
    function = scipy.interpolate.interp1d(model.soc_vector[index], model.ocv_vector[index])
    return function(soc)


def optfn(theGParam, data, model, temperature, doHyst):
    """
    Optimization function copied from the Octave code
    """
    global bestcost
    bestcost = np.inf
    model.GParam.append(abs(theGParam))
    [cost, _] = minfn(data, model, temperature, doHyst)
    if cost < bestcost:
        bestcost = cost
        print("The model created for this value of gamma is the best ESC model yet!")
    return cost


def SISOsubid(y, u, n):
    """
    Identifies state-space "A" matrix from input-output data.
       y: vector of measured outputs
       u: vector of measured inputs
       n: number of poles in solution

       A: discrete-time state-space state-transition matrix.

    Theory from "Subspace Identification for Linear Systems
                 Theory - Implementation - Applications"
                 Peter Van Overschee / Bart De Moor (VODM)
                 Kluwer Academic Publishers, 1996
                 Combined algorithm: Figure 4.8 page 131 (robust)
                 Robust implementation: Figure 6.1 page 169

    Code adapted from "subid.m" in "Subspace Identification for
                 Linear Systems" toolbox on MATLAB CENTRAL file
                 exchange, originally by Peter Van Overschee, Dec. 1995
    """
    # inputs looks very precise (u seems completely same as in Octave)
    ny = len(y)
    y = np.array(y)

    nu = len(u)
    u = np.array(u)

    i = 2 * n  # two times the number of poles
    twoi = 4 * n
    if nu != ny:
        print("nu and ny must be the same size")
        raise Exception()
    if (ny - twoi + 1) < twoi:
        print("Not enough data points")
        raise Exception()

    # Determine the number of columns in the Hankel matrices
    j = ny - twoi + 1

    # Make Hankel matrices Y and U
    Y = np.zeros((twoi, j))
    U = np.zeros((twoi, j))
    for k in range(twoi):
        Y[k, :] =y[k : k + j]
        U[k, :] =u[k : k + j]
    # U looks the same and Y is roughly 99% same as results in octave code
    # Compute the R factor
    Rtemp = np.concatenate((U, Y))
    Rtemp = Rtemp.transpose()
    R = LA.qr(Rtemp)[1].transpose()
    # This looks a little cleaner than in Octave but that is because functions work differently
    # Still, the end result for R looks similar to what we get in Octave

    # First step: Calculate oblique and orthogonal projections
    Rf = R[3 * i: 4 * i, :]  # Future outputs
    Rp = np.append(R[0:i, :], R[2 * i: 3 * i, :], 0)  # Past inputs and outputs
    Ru = R[1 * i:2 * i, 0: twoi]  # Future inputs
    # Perpendicular future outputs

    Rfp_var = np.dot(Rf[:, 0:twoi], np.linalg.pinv(Ru))
    Rfp_var1 = np.dot(Rfp_var, Ru)
    Rfp = np.append(Rf[:, 0: twoi] - Rfp_var1, Rf[:, twoi: 4 * i], 1)
    # Perpendicular past inputs and outputs
    Rpp_var = np.dot(Rp[:, 0:twoi], np.linalg.pinv(Ru))
    Rpp_var1 = np.dot(Rpp_var, Ru)
    Rpp = np.append(Rp[:, 0: twoi] - Rpp_var1, Rp[:, twoi: 4 * i], 1)

    # The oblique projection is computed as (6.1) in VODM, page 166.
    # obl / Ufp = Yf / Ufp * pinv(Wp / Ufp) * (Wp / Ufp)
    # The extra projection on Ufp(Uf perpendicular) tends to give better numerical conditioning
    # (see algo on VODM page 131)

    # This rank check is needed to avoid rank deficiency warnings
    if LA.norm(Rpp[:, 3 * i-2:3 * i], 'fro') < 1e-10:
        Ob = (Rfp * LA.pinv(Rpp.transpose())).transpose() * Rp  # Oblique projection
    else:
        # Ob = (Rfp / Rpp) * Rp
        Ob_var = np.dot(Rfp, np.linalg.pinv(Rpp))
        Ob = np.dot(Ob_var, Rp)

    # Second step: Compute weighted oblique projection and its SVD
    # Extra projection of Ob on Uf perpendicular
    WOW_var = np.dot(Ob[:, 0:twoi], np.linalg.pinv(Ru))
    WOW_var1 = np.dot(WOW_var, Ru)
    WOW = np.append(Ob[:, 0: twoi] - WOW_var1, Ob[:, twoi: 4 * i], 1)
    U, S, _ = LA.svd(WOW)
    ss = np.transpose(S)

    # Third step: Partitioning U into U1 and U2 (the latter is not used)
    U1 = U[:, 0: n]  # Determine U1

    # Fourth step: Determine gam and gamm
    gam = U1 * np.diag(np.sqrt(ss[0:n]))
    gamm = gam[0:(i - 1), :]
    gam_inv = LA.pinv(gam)  # Pseudo inverse of gam
    gamm_inv = LA.pinv(gamm)  # Pseudo inverse of gamm

    # Fifth step: Determine A matrix(also C, which is not used)
    Rhs_var = np.append(np.dot(gam_inv, R[3 * i:4 * i, 0: 3 * i]), np.zeros((n, 1)), 1)
    Rhs = np.append(Rhs_var, R[i: twoi, 0: 3 * i + 1], 0)
    Lhs = np.append(np.dot(gamm_inv, R[3 * i + 1:4 * i, 0: 3 * i + 1]), R[3 * i : 3 * i + 1, 0: 3 * i + 1], 0)
    sol = np.dot(Lhs, LA.pinv(Rhs))  # (sol = Lhs / Rhs) Solve least squares for [A;C]
    A = sol[0:n, 0: n]  # Extract A

    return A


def minfn(data, model, temperature, doHyst):
    """
    Minimization function
    """
    alltemps = [data[i].temp for i in range(len(data))]
    index_array = np.where(np.array(alltemps) == temperature)
    ind = index_array[0][0]
    numfiles = len(index_array)  # will be 1 for now

    xplots = np.ceil(np.sqrt(numfiles))
    yplots = np.ceil(numfiles / xplots)
    rmserr = np.zeros(1, xplots * yplots)

    G = abs(model.GParam[ind])      # for 25 degrees
    Q = abs(model.QParam[ind])      # for 25 degrees
    eta = abs(model.etaParam[ind])  # for 25 degrees
    # RC = model.RCParam[ind]       # for 25 degrees
    numpoles = 1  # len(RC)         # for 25 degrees # todo extend this functionality

    for thefile in range(numfiles):  # should always be 1 file as long as there is one test per temperature
        ik = data[ind].script1.current[:]
        vk = data[ind].script1.voltage[:]
        # tk = range(len(vk))  # never used again
        etaik = ik
        for i in range(len(etaik)):
            if etaik[i] < 0:
                etaik[i] = etaik[i] * eta

        h = [0] * len(ik)
        sik = [0] * len(ik)
        fac = np.exp(-abs(G * np.array(etaik) / (3600 * Q)))  # looks the same as octave
        for k in range(1, len(ik)):
            h[k] = fac[k - 1] * h[k - 1] + (fac[k - 1] - 1) * np.sign(ik[k - 1])
            sik[k] = np.sign(ik[k])
            if abs(ik[k]) < Q / 100:
                sik[k] = sik[k-1]

        # First modeling step: Compute error with model = OCV only
        # Everything looks the same up until this part (except OCV and SOC arrays)
        vest1 = data[ind].OCV  # OCV is not completely the same but vk is
        v_error = np.array(vk) - np.array(vest1)  # therefore v_error is not the same as in Octave
        numpoles_loop_no = numpoles

        # Second modeling step: Compute time constants in "A" matrix
        while True:
            A = SISOsubid(-np.diff(v_error), np.diff(etaik), numpoles_loop_no)   # diff works fine
            eigA = LA.eig(A)[0]
            assert (eigA == np.conj(eigA)), "eigA is not a real number"
            assert (1 > eigA > 0), "eigA is not in proper range"
            okpoles = len(eigA)
            numpoles_loop_no = numpoles_loop_no + 1
            if okpoles >= numpoles:
                break
            print(f'Trying {numpoles_loop_no=}\n')

        RCfact_var = np.sort(eigA)
        RCfact = RCfact_var[len(RCfact_var) - numpoles:]
        RC = -1 / np.log(RCfact)  # reference code says 2.3844, but we get slightly over 2.4
        # Simulate the R - C filters to find R - C currents
        vrcRaw = np.zeros((numpoles, len(h)))
        for k in range(1, len(ik)):
            vrcRaw[:, k] = np.diag(RCfact) * vrcRaw[:, k - 1] + (1 - RCfact) * etaik[k - 1]
        vrcRaw = np.transpose(vrcRaw)  # Close enough to Octave vrcRaw

        # Third modeling step: Hysteresis parameters
        if doHyst:
            H_1 = np.append(np.transpose([h]), np.transpose([sik]), 1)
            H_2 = np.append(np.transpose([-etaik]), -vrcRaw, 1)
            H = np.append(H_1, H_2, 1)
            W = LA.lstsq(H, v_error)  # W = H\verr;   LEAST SQUARE NON NEGATIVE
            M  = W[0][0]
            M0 = W[0][1]
            R0 = W[0][2]
            Rfact = np.transpose(W[0][3:])
        else:
            H = np.append(np.transpose([-etaik]), -vrcRaw, 1)
            W = H / v_error  # Todo probably something wrong here, revisit later
            M = 0
            M0 = 0
            R0 = W[0]
            Rfact = np.transpose(W[0][1:])

        # Populate the model
        model.R0Param[ind] = R0
        model.M0Param[ind] = M0
        model.MParam[ind] = M
        model.RCParam[ind] = np.transpose(RC)
        model.RParam[ind] = np.transpose(Rfact)

        vest2 = vest1 + np.array(h)*M + M0*np.array(sik) - R0*np.array(etaik) - np.transpose(vrcRaw) * Rfact
        verr = vk - vest2
        # starting to look different

        # Compute RMS error only on data roughly in 5 % to 95 % SOC
        v1 = OCVfromSOCtemp(0.95, data[ind].temp, model)
        v2 = OCVfromSOCtemp(0.05, data[ind].temp, model)
        N1_array = np.where(vk < v1, 1, 0)
        N2_array = np.where(vk < v2, 1, 0)
        for i in range(len(N1_array)):
            if N1_array[i] == 1:
                N1 = i
                break
        for i in range(len(N2_array)):
            if N2_array[i] == 1:
                N2 = i
                break
        if not N1:
            N1=1
        if not N2:
            N2=len(verr)
        rmserr[thefile] = np.sqrt(np.mean(verr[0, N1:N2]**2))

    cost = sum(rmserr)
    print(f'RMS error for present value of gamma = {cost * 1000} (mV)\n')
    assert cost, 'Exception: Cost is empty'

    return [cost, model]

