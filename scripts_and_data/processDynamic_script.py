import numpy as np
from numpy import linalg as LA
import scipy
from scipy import optimize
fminbnd = scipy.optimize.fminbound
import generate_battery_cell_data
import matplotlib.pyplot as plt


current_sign_threshold = 1e-8  # threshold for current sign (in octave it is Q/100 and this gives me the negative M0)


def processDynamic(dynamic_data, model, numpoles, doHyst):
    """
    Script that populates the model parameters based on dynamic test data
    """
    # First step: Compute Q and eta again. This is to recalibrate the static test Q and eta
    ind25 = None
    for index, single_temp_data in enumerate(dynamic_data):
        if single_temp_data.temp == 25:
            ind25 = index
    if not ind25:
        print("There is no default temperature of 25 deg celsius!")
        raise Exception()

    totDisAh = dynamic_data[ind25].script1.disAh[-1] + dynamic_data[ind25].script2.disAh[-1] + dynamic_data[ind25].script3.disAh[-1]
    totChgAh = dynamic_data[ind25].script1.chgAh[-1] + dynamic_data[ind25].script2.chgAh[-1] + dynamic_data[ind25].script3.chgAh[-1]
    eta25 = totDisAh / totChgAh

    dynamic_data[ind25].script1.chgAh[-1] = eta25 * dynamic_data[ind25].script1.chgAh[-1]  # correct for the coefficients
    dynamic_data[ind25].script2.chgAh[-1] = eta25 * dynamic_data[ind25].script2.chgAh[-1]  # correct for the coefficients
    dynamic_data[ind25].script3.chgAh[-1] = eta25 * dynamic_data[ind25].script3.chgAh[-1]  # correct for the coefficients
    Q25 = dynamic_data[ind25].script1.disAh[-1] + dynamic_data[ind25].script2.disAh[-1] - dynamic_data[ind25].script1.chgAh[-1] - dynamic_data[ind25].script2.chgAh[-1]
    dynamic_data[ind25].Q = Q25
    dynamic_data[ind25].eta = eta25

    for k in range(len(dynamic_data)):
        if dynamic_data[k].temp != 25:
            dynamic_data[k].script2.chgAh = dynamic_data[k].script2.chgAh * eta25
            dynamic_data[k].script3.chgAh = dynamic_data[k].script3.chgAh * eta25
            eta = (dynamic_data[k].script1.disAh[-1] + dynamic_data[k].script2.disAh[-1] +
                   dynamic_data[k].script3.disAh[-1] - dynamic_data[k].script2.chgAh[-1] -
                   dynamic_data[k].script3.chgAh[-1]) / dynamic_data[k].script1.chgAh[-1]

            dynamic_data[k].script1.chgAh = eta * dynamic_data[k].script1.chgAh
            Q = dynamic_data[k].script1.disAh[-1] + dynamic_data[k].script2.disAh[-1] - dynamic_data[k].script1.chgAh[-1] - dynamic_data[k].script2.chgAh[-1]
            dynamic_data[k].Q = Q
            dynamic_data[k].eta = eta

        # Populate model with parameters from the dynamic data
        model.QParam.append(dynamic_data[k].Q)
        model.etaParam.append(dynamic_data[k].eta)
        if model.temps[k] != dynamic_data[k].temp:
            print("Dynamic data temperatures are not matching the static data temperatures or their numbering order.")
            raise Exception()

        # Todo: Delete this after testing
        # model.QParam[k]  = model.QParam_static[k]
        # model.etaParam[k]= model.etaParam_static[k]

    # Second step: Compute OCV for "discharge portion" of test, based on static test data
    for k in range(len(dynamic_data)):
        corrected_current = [0]*len(dynamic_data[k].script1.current)
        for index, current in enumerate(dynamic_data[k].script1.current):
            if current < 0:  # if current is flowing into the battery and charging it, apply a coefficient
                corrected_current[index] = current * model.etaParam[k]
            else:
                corrected_current[index] = current
        dynamic_data[k].Z = np.ones(np.size(corrected_current)) - np.cumsum(corrected_current)/(dynamic_data[k].Q*3600)
        dynamic_data[k].OCV = OCVfromSOCtemp(dynamic_data[k].Z, dynamic_data[k].temp, model)
        # plt.plot(data[k].script1.time, data[k].Z)

    # Third step: Use optimization algorythm to find parameters M, M0, G, RC, R and R0
    model.GParam = [0] * len(dynamic_data)
    model.MParam = [0] * len(dynamic_data)
    model.M0Param = [0] * len(dynamic_data)
    model.RParam = [0] * len(dynamic_data)
    model.R0Param = [0] * len(dynamic_data)
    model.RCParam = [0] * len(dynamic_data)
    model.CParam = [0] * len(dynamic_data)

    for k in range(len(dynamic_data)):
        global bestcost
        bestcost = np.inf
        print("Processing temperature: ", dynamic_data[k].temp, " degrees celsius")
        if doHyst:
            model.GParam[k] = abs(fminbnd(optfn, 1, 250, args=(dynamic_data, model, model.temps[k], doHyst, k), xtol=0.1, maxfun=40, disp=0))
        else:  # Todo check functionality and extend it if it doesnt work
            model.GParam[k] = 0
            optfn(0, dynamic_data, model, model.temps[k], doHyst, k)

        # Todo I dont think this last thing here is even needed
        [_, model] = minfn(dynamic_data, model, model.temps[k], doHyst)

    print("Dynamic model created!")
    return model


def OCVfromSOCtemp(soc, temp, model):
    """
    Extrapolates OCV vector from the soc vector but based on the previous static OCV/SOC test contained in the model.
    """
    index = np.where(np.array(model.temps) == temp)[0][0]
    function = scipy.interpolate.interp1d(model.soc_vector[index], model.ocv_vector[index], fill_value="extrapolate")
    return function(soc)


def optfn(theGParam, dynamic_data, model, temperature, doHyst, index):
    """
    Optimization function copied from the Octave code
    # Todo: Fuse with minfn as it doesn't have to be separated like in octave
    """
    print("optfn function was triggered")
    global bestcost
    model.GParam[index] = abs(theGParam)
    [cost, _] = minfn(dynamic_data, model, temperature, doHyst)
    if cost < bestcost:
        bestcost = cost
        print("The model created for this value of gamma is the best ESC model yet!")
    return cost


def minfn(dynamic_data, model, temperature, doHyst):
    """
    Minimization function
    """
    print("minfn function was triggered")
    alltemps = [dynamic_data[i].temp for i in range(len(dynamic_data))]
    index_array = np.where(np.array(alltemps) == temperature)
    ind = index_array[0][0]  # index of current temperature in the temperatures vector of model (and data?)
    numfiles = len(index_array)  # will be 1 for now

    xplots = np.ceil(np.sqrt(numfiles))
    yplots = np.ceil(numfiles / xplots)
    root_mean_square_error = np.zeros(1, xplots * yplots)

    G = model.GParam[ind]             # Boulder data: temp_5 -> G = 96.10954, 154.89
    Q = model.QParam[ind]             # Boulder data: temp_5 -> Q = 14.5924882
    eta = model.etaParam[ind]         # Boulder data: temp_5 -> eta = 0.981744
    # RC = model.RCParam[ind]
    numpoles = 1  # len(RC)          # todo extend this functionality

    for thefile in range(numfiles):  # should always be 1 file as long as there is one test per temperature
        script_1_current = dynamic_data[ind].script1.current[:]
        script_1_voltage = dynamic_data[ind].script1.voltage[:]
        # tk = range(len(vk))  # never used again
        script_1_current_corrected = script_1_current  # Todo Make this not a copy
        for i in range(len(script_1_current)):
            if script_1_current[i] < 0:
                script_1_current_corrected[i] = script_1_current[i] * eta
            else:
                script_1_current_corrected[i] = script_1_current[i]

        # Calculating hysteresis variables
        h = [0] * len(script_1_current)  # 'h' is a variable that relates to hysteresis voltage
        current_sign = [0] * len(script_1_current)
        # time step between two calculated current points
        delta_T = generate_battery_cell_data.SIMULATION_SPEED_UP*generate_battery_cell_data.Ts
        fac = np.exp(-abs(G * np.array(script_1_current_corrected) / (3600 * Q) * delta_T))  # also a hysteresis voltage variable
        # debug looks the same as octave up until this point
        for k in range(1, len(script_1_current)):  # todo check why it starts with 1
            h[k] = fac[k - 1] * h[k - 1] + (fac[k - 1] - 1) * np.sign(script_1_current[k - 1])
            current_sign[k] = np.sign(script_1_current[k])
            if abs(script_1_current[k]) < current_sign_threshold:  # threshold for current sign
                current_sign[k] = current_sign[k-1]

        # First modeling step: Compute error with model represented only with OCV
        # debug looks the same as octave up until this point (except OCV and SOC arrays)
        v_est_ocv = dynamic_data[ind].OCV  # OCV is not completely the same but script_1_voltage is the same
        v_error = np.array(script_1_voltage) - np.array(v_est_ocv)  # therefore v_error is not the same as in Octave
        # v_error = [-0.001688.....] for Boulder data
        numpoles_loop_no = numpoles

        # Second modeling step: Compute time constants in "A" matrix, or in other terms, RC circuit parameters
        # Octave code was not up to date with lessons from here
        # diff works fine
        # https://www.coursera.org/lecture/equivalent-circuit-cell-model-simulation/2-3-3-introducing-octave-code-to-determine-dynamic-part-of-an-ecm-NILTD
        while True:
            A = SISOsubid(-np.diff(v_error), np.diff(script_1_current_corrected), numpoles_loop_no)
            eigA = LA.eig(A)[0]     # For Boulder: eigA = [0.2389149], [0.6528]
            if eigA != np.conj(eigA):
                print("WARNING: eigA is not a real number, results may not be proper. eigA = ", eigA)
                eigA = abs(eigA)
            if not (1 > eigA > 0):
                print("WARNING: eigA is not in proper range, results may not be proper. eigA = ", eigA)
                if eigA < 0:
                    eigA = -eigA
                elif eigA > 1:
                    eigA = 1/eigA
            okpoles = len(eigA)
            numpoles_loop_no = numpoles_loop_no + 1
            if okpoles >= numpoles:
                break
            print(f'Trying {numpoles_loop_no=}\n')

        RCfact_var = np.sort(eigA)
        RCfact = RCfact_var[len(RCfact_var) - numpoles:]
        RC = -1 / np.log(RCfact)  # reference code says 2.3844, but we get slightly over 2.4
        # Simulate the R - C filters to find R - C currents
        resistor_current_rc = np.zeros((numpoles, len(h)))
        for k in range(1, len(script_1_current)):
            resistor_current_rc[:, k] = np.diag(RCfact) * resistor_current_rc[:, k - 1] + (1 - RCfact) * script_1_current_corrected[k - 1]
        resistor_current_rc = np.transpose(resistor_current_rc)  # Close enough to Octave vrcRaw

        # Third modeling step: Hysteresis parameters
        if doHyst:
            H_1 = np.append(np.transpose([h]), np.transpose([current_sign]), 1)
            H_2 = np.append(np.transpose([-script_1_current_corrected]), -resistor_current_rc, 1)
            H = np.append(H_1, H_2, 1)
            W = LA.lstsq(H, v_error)  # W = H\verr;   LEAST SQUARE NON NEGATIVE
            M  = W[0][0]
            M0 = W[0][1]
            R0 = W[0][2]
            Rfact = np.transpose(W[0][3:])  # rest of the lstsq array values
        else:
            H = np.append(np.transpose([-script_1_current_corrected]), -resistor_current_rc, 1)
            W = LA.lstsq(H, v_error)
            M = 0
            M0 = 0
            R0 = W[0][0]
            Rfact = np.transpose(W[0][1:])  # rest of the lstsq array values

        # Populate the model
        model.R0Param[ind] = R0
        model.M0Param[ind] = M0
        model.MParam[ind] = M
        model.RCParam[ind] = np.transpose(RC)[0]
        model.RParam[ind] = np.transpose(Rfact)[0]
        model.CParam[ind] = model.RCParam[ind]/model.RParam[ind]

        v_est_full = v_est_ocv + np.array(h)*M + M0*np.array(current_sign) - R0*np.array(script_1_current_corrected) - np.transpose(vrcRaw) * Rfact
        verr = script_1_voltage - v_est_full
        # starting to look different

        # Compute RMS error only on data roughly in 5 % to 95 % SOC
        v1 = OCVfromSOCtemp(0.95, dynamic_data[ind].temp, model)
        v2 = OCVfromSOCtemp(0.05, dynamic_data[ind].temp, model)
        N1_array = np.where(script_1_voltage < v1, 1, 0)
        N2_array = np.where(script_1_voltage < v2, 1, 0)
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
        root_mean_square_error[thefile] = np.sqrt(np.mean(verr[0, N1:N2]**2))

    cost = sum(root_mean_square_error)
    print(f'RMS error for present value of gamma = {cost * 1000} (mV)\n')
    assert cost, 'Exception: Cost is empty'

    return [cost, model]


def SISOsubid(y, u, n):  # Subspace system identification function
    """
    This function calculates values for time constants of RC circuit, based on the number of poles of the system.
    The solutions may not always be between 0 and 1, they may also be negative or even complex conjugate.
    In these cases, we warn the user but still provide the time constants.

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
