import numpy as np
from numpy import linalg as LA
import scipy
from scipy import optimize
fminbnd = scipy.optimize.fminbound
import generate_battery_cell_data as cell_data
import generate_pickled_cell_model as cell_model
import battery_cell_functions as cell_functions
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook
import pandas as pd
import os


def processDynamic(dynamic_data, model, numpoles, doHyst, typhoon_origin=False):
    """
    Script that populates the model parameters (R0, R1 - C1 pairs, M0, M, Gamma) based on dynamic test data
    """
    excel_data, row_index = [], []  # for result compilation
    model.GParam = [0] * len(dynamic_data)
    model.MParam = [0] * len(dynamic_data)
    model.M0Param = [0] * len(dynamic_data)
    model.RParam = [0] * len(dynamic_data)
    model.R0Param = [0] * len(dynamic_data)
    model.RCParam = [0] * len(dynamic_data)
    model.CParam = [0] * len(dynamic_data)

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

    for k in range(len(dynamic_data)):
        if dynamic_data[k].temp != 25:
            dynamic_data[k].script2.chgAh = dynamic_data[k].script2.chgAh * eta25
            dynamic_data[k].script3.chgAh = dynamic_data[k].script3.chgAh * eta25
            eta = (dynamic_data[k].script1.disAh[-1] + dynamic_data[k].script2.disAh[-1] +
                   dynamic_data[k].script3.disAh[-1] - dynamic_data[k].script2.chgAh[-1] -
                   dynamic_data[k].script3.chgAh[-1]) / dynamic_data[k].script1.chgAh[-1]

            dynamic_data[k].script1.chgAh = eta * dynamic_data[k].script1.chgAh
            Q = dynamic_data[k].script1.disAh[-1] + dynamic_data[k].script2.disAh[-1] - dynamic_data[k].script1.chgAh[-1] - dynamic_data[k].script2.chgAh[-1]
        else:
            Q = Q25
            eta = eta25

        # Populate model with parameters from the dynamic data
        model.QParam.append(model.QParam_static[k] if cell_model.use_static_Q_eta else Q)
        model.etaParam.append(model.etaParam_static[k] if cell_model.use_static_Q_eta else eta)

        if model.temps[k] != dynamic_data[k].temp:
            print("Dynamic data temperatures are not matching the static data temperatures or their numbering order.")
            raise Exception()

        # Second step: Compute OCV for "discharge portion" of test, based on static test data
        corrected_current = [0]*len(dynamic_data[k].script1.current)
        for index, current in enumerate(dynamic_data[k].script1.current):
            if current < 0:  # if current is flowing into the battery and charging it, apply a coefficient
                corrected_current[index] = current * model.etaParam[k]
            else:
                corrected_current[index] = current
        dynamic_data[k].Z = np.ones(np.size(corrected_current)) - np.cumsum(corrected_current)/(model.QParam[k]*3600)
        dynamic_data[k].OCV = OCVfromSOCtemp(dynamic_data[k].Z, dynamic_data[k].temp, model)

        OCV_error = (dynamic_data[k].script1.OCV_real - dynamic_data[k].OCV)
        cell_functions.plot_func([dynamic_data[k].script1.time], [OCV_error],
                       [f"OCV real vs OCV calculated through time, calculated for dynamic script 1 "
                        f"(discharging) for temp = {dynamic_data[k].temp}"],
                       flag_show=False)

        # Third step: Use optimization algorithm to find parameters M, M0, G, RC, R and R0
        print("Processing temperature: ", dynamic_data[k].temp, " degrees celsius")
        if doHyst:
            GParam_optimal = fminbnd(minfn, 1, 250, args=(dynamic_data, model, model.temps[k], doHyst, typhoon_origin), xtol=0.1, maxfun=40, disp=2)
            print(f"Converged value of GParam is {round(GParam_optimal)}")
        else:
            minfn(0, dynamic_data, model, model.temps[k], doHyst, typhoon_origin)

        # Data needed to determine RC circuits (poles) is saved for external processing as well as excel data for visualizing
        SISOSubid_data = {'verr': np.array(dynamic_data[k].script1.voltage) - np.array(dynamic_data[k].OCV),
                          'curr_corrected': corrected_current,
                          'curr': dynamic_data[k].script1.current}
        scipy.io.savemat("SUB_ID.mat", SISOSubid_data)

        V_resistor = model.R0Param[k]*np.array(dynamic_data[k].script1.current)
        rc_current = get_rc_current(dynamic_data[k].script1.current, delta_T=1, RC1=model.RCParam[k], discretization="euler")
        V_diff = np.array(rc_current) * model.RParam[k]
        h = get_h_list(dynamic_data[k].script1.current, model.GParam[k], model.QParam[k], eta=model.etaParam[k], delta_T=1)
        V_h = np.array(h) * model.MParam[k] + np.sign(dynamic_data[k].script1.current) * model.M0Param[k]
        err = np.array(dynamic_data[k].script1.voltage) - (np.array(dynamic_data[k].OCV) + V_resistor + V_diff + V_h)

        excel_data.append(np.array(dynamic_data[k].script1.voltage))  # reference dynamic voltage data
        excel_data.append(np.array(dynamic_data[k].OCV) + V_resistor + V_diff + V_h)  # generated dynamic voltage data
        excel_data.append(dynamic_data[k].OCV)  # generated OCV data
        excel_data.append(V_resistor)  # generated internal resistance voltage drop
        excel_data.append(V_diff)  # generated diffusion voltage drop
        excel_data.append(V_h)  # generated hysteresis voltage drop
        excel_data.append(err)  # generated error between reference and results
        excel_data.append([sum(err**2)] * len(err))  # RMS error combined, but has to be of certain length
        row_index.append(f'{model.temps[k]} reference dynamic voltage data')
        row_index.append(f'{model.temps[k]} generated dynamic voltage data')
        row_index.append(f'{model.temps[k]} generated OCV data')
        row_index.append(f'{model.temps[k]} generated internal resistance voltage drop')
        row_index.append(f'{model.temps[k]} generated diffusion voltage drop')
        row_index.append(f'{model.temps[k]} generated hysteresis voltage drop')
        row_index.append('Error calculation')
        row_index.append('Total RMS')

    print("Dynamic model created!")

    df = pd.DataFrame(data=excel_data,
                      index=row_index)
    df = df.T
    df.to_excel(cell_model.data_origin + ".xlsx")
    print("Excel data saved as ", cell_model.data_origin + ".xlsx")
    os.system("start EXCEL.EXE " + cell_model.data_origin + ".xlsx")

    return model


def minfn(G, dynamic_data, model, temperature, doHyst, typhoon_origin, numpoles=1):
    """
    Minimization function
    """
    print("minfn function was triggered")
    alltemps = [dynamic_data[i].temp for i in range(len(dynamic_data))]
    index_array = np.where(np.array(alltemps) == temperature)
    ind = index_array[0][0]  # index of current temperature in the temperatures vector of model (and data?)

    model.GParam[ind] = G             # Boulder data: temp_5 -> G = 96.10954, 154.89
    Q = model.QParam[ind]             # Boulder data: temp_5 -> Q = 14.5924882
    eta = model.etaParam[ind]         # Boulder data: temp_5 -> eta = 0.981744
    # RC = model.RCParam[ind]
    # numpoles = len(RC)                # todo extend this functionality

    script_1_current = dynamic_data[ind].script1.current[:]
    current_sign = np.sign(script_1_current)
    script_1_voltage = dynamic_data[ind].script1.voltage[:]
    script_1_current_corrected = np.copy(script_1_current)
    for i in range(len(script_1_current)):
        if script_1_current[i] < 0:
            script_1_current_corrected[i] = script_1_current[i] * eta
        else:
            script_1_current_corrected[i] = script_1_current[i]

    # Calculating hysteresis variables
    # time step between two calculated current points
    if typhoon_origin:
        delta_T = cell_data.SIMULATION_SPEED_UP*cell_data.Ts
        delta_T_hyst = cell_data.SIMULATION_SPEED_UP*cell_data.Ts_cell
    else:
        delta_T = 1  # For Colorado Boulder, delta T for samples is 1 second (at least for dynamic data)
        delta_T_hyst = 1  # For Colorado Boulder, delta T for samples is 1 second (at least for dynamic data)

    h = get_h_list(script_1_current, G, Q, eta, delta_T)

    # First modeling step: Compute error with model represented only with OCV
    # debug looks the same as octave up until this point (except OCV and SOC arrays)
    v_est_ocv = dynamic_data[ind].OCV  # OCV is not completely the same but script_1_voltage is the same
    v_error = np.array(script_1_voltage) - np.array(v_est_ocv)  # therefore v_error is not the same as in Octave
    # v_error = [-0.001688.....] for Boulder data
    numpoles_loop_no = numpoles

    # Second modeling step: Compute time constants in "A" matrix, or in other terms, RC circuit parameters
    if cell_model.minimization == "double_minimize" or cell_model.minimization == "differential_evolution":

        bnds = ((0, 1), (0, 1), (5, 120))  # Bounds for minimization functions  R0, R1, RC1
        if cell_model.minimization == "differential_evolution":
            params = optimize.differential_evolution(double_minimization, args=(v_error, -script_1_current, delta_T),
                                                     bounds=bnds)
        else:  # double_minimize
            # Todo: init_guess is fitting the curve too much
            # Todo: add R0 guess from static test (maybe even RC and R1)
            init_guess = [cell_data.R0Param[ind], cell_data.Rparam[ind], cell_data.RCparam[ind]]  # R0, R1, RC1
            # init_guess = [4e-3, 3e-3, 120]  # R0, R1, RC1
            params = optimize.minimize(double_minimization, init_guess, method="Nelder-Mead",
                                       args=(v_error, -script_1_current, delta_T), bounds=bnds, tol=1e-6)
            """
            Nelder-Mead     # actually good
            Powell          # actually good
            """

        R0 = params.x[0]
        R1 = params.x[1]
        RC1 = params.x[2]
        model.R0Param[ind] = R0
        model.RParam[ind] = R1
        model.RCParam[ind] = RC1
        model.CParam[ind] = RC1/R1

        # Initialize RC resistor current for error calculation
        resistor_current_rc = get_rc_current(script_1_current, delta_T=delta_T, RC1=RC1, discretization="euler")

        # Third modeling step: Hysteresis parameters
        if doHyst:
            # Todo -current_sign when I finally update the battery cell
            H = np.append(np.transpose([h]), np.transpose([current_sign]), 1)
            W = LA.lstsq(H, v_error)  # finds best W for H@W=v_error
            model.MParam[ind] = W[0][0]
            model.M0Param[ind] = W[0][1]
        else:
            model.M0Param[ind] = 0
            model.MParam[ind] = 0

        v_est_full = v_est_ocv + np.array(h) * model.MParam[ind] + model.M0Param[ind] * current_sign - \
                     R0 * np.array(script_1_current) - np.array(resistor_current_rc) * R1

    else:
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

        # Solution of SISOSubid is RCfact which is np.exp(-delta_T/Tau) and as exponential function, the solution should be between 0 and 1
        RCfact_var = np.sort(eigA)  # [0.66075] for boulder data
        RCfact = (RCfact_var[len(RCfact_var) - numpoles:])[0]  # looks like it makes no sense to be different then previous variable
        RC = (-1 / np.log(RCfact))[0]  # reference code says 2.3844, but we get slightly over 2.4 s (or minutes)
        # 1 in previous function is delta_T and should be a variable but then that must also be propagated in SISOSubid
        # Simulate the R - C filters to find R - C currents
        RC = 60
        resistor_current_rc = get_rc_current(script_1_current, discretization="euler", numpoles=numpoles, RCfact=RCfact, RC1=RC)

        # Third modeling step: Hysteresis parameters
        if doHyst:
            # Todo: shorten into one line
            if typhoon_origin:
                H_1 = np.append(np.transpose([h]), -np.transpose([current_sign]), 1)  # in typhoon current sign is opposite
                H_2 = np.append(-np.transpose([script_1_current]), -np.transpose([resistor_current_rc]), 1)  # In typhoon, correction coefficient is not applied to current, only to SOC
            else:
                H_1 = np.append(np.transpose([h]), np.transpose([current_sign]), 1)
                H_2 = np.append(np.transpose([-script_1_current_corrected]), -np.transpose([resistor_current_rc]), 1)
            H = np.append(H_1, H_2, 1)
            W = LA.lstsq(H, v_error)  # finds best W for H@W=v_error
            M  = W[0][0]
            M0 = W[0][1]
            R0 = W[0][2]
            R1 = np.transpose(W[0][3:])  # rest of the lstsq array values
        else:
            if typhoon_origin:  # In typhoon, correction coefficient is not applied to current, only to SOC
                H = np.append(np.transpose([-script_1_current]), -resistor_current_rc, 1)
            else:
                H = np.append(np.transpose([-script_1_current_corrected]), -resistor_current_rc, 1)
            W = LA.lstsq(H, v_error)  # finds best W for H@W=v_error
            M = 0
            M0 = 0
            R0 = W[0][0]
            R1 = np.transpose(W[0][1:])  # rest of the lstsq array values

        # Populate the model
        model.RCParam[ind] = RC
        model.RParam[ind] = np.transpose(R1)[0]
        model.CParam[ind] = model.RCParam[ind]/model.RParam[ind]
        model.R0Param[ind] = R0
        model.M0Param[ind] = M0
        model.MParam[ind] = M

        v_est_full = v_est_ocv + np.array(h) * M + M0 * np.array(current_sign) - R0 * np.array(script_1_current_corrected) - np.transpose(resistor_current_rc) * R1

    verr = script_1_voltage - v_est_full  # v_est_full represents fully estimated data

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
        N1 = 1
    if not N2:
        N2 = len(verr)

    root_mean_square_error = np.sqrt(np.mean(verr[N1:N2]**2))
    print(f'RMS error for present value of gamma = {root_mean_square_error * 1000} (mV)\n')
    assert root_mean_square_error, 'Exception: Cost is empty'

    return root_mean_square_error


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


def double_minimization(params, verr, curr, Ts):
    """
    Minimization function that should find the best values for R1, C1 and R0
    Formula is derived using forward euler discretization method, similar to how it is implemented in software
    """
    R0=params[0]
    R1=params[1]
    RC1=params[2]
    verr_calculated = [0]*len(curr)
    verr_calculated[0] = verr[0]
    error = 0
    Tau_Ts = RC1/Ts
    for k in range(1, len(curr)):  # start from index 1
        # verr_calculated[k] = curr[k]*(R0*R1*C1+R0*Ts+R1*Ts)/(Ts+R1*C1) - curr[k-1]*(R0*R1*C1)/(Ts+R1*C1) + verr_calculated[k-1]*(R1*C1)/(Ts+R1*C1)

        verr_calculated[k] = Tau_Ts*verr_calculated[k-1] + curr[k]*(R0+R1+Tau_Ts*R0) - curr[k-1]*(Tau_Ts*R0)
        verr_calculated[k] = verr_calculated[k]/(1+Tau_Ts)

        error += (verr[k]-verr_calculated[k])**2  # RMS error calculation
    error = error/len(curr)

    return error


def get_h_list(script_1_current, G, Q, eta, delta_T=1):
    """
    Returns list of h values which in conjuction with M and M0 params creates a hysteresis voltage drop
    """
    current_sign = np.sign(script_1_current)
    script_1_current_corrected = np.copy(script_1_current)
    for i in range(len(script_1_current)):
        if script_1_current[i] < 0:
            script_1_current_corrected[i] = script_1_current[i] * eta
        else:
            script_1_current_corrected[i] = script_1_current[i]

    h = [0] * len(script_1_current)  # 'h' is a variable that relates to hysteresis voltage
    fac = np.exp(-abs(G * script_1_current_corrected / (3600 * Q) * delta_T))  # also a hysteresis voltage variable
    for k in range(1, len(script_1_current)):
        h[k] = fac[k - 1] * h[k - 1] + (fac[k - 1] - 1) * current_sign[k - 1]

    return h


def get_rc_current(script_1_current, delta_T=1, RC1=None, discretization="euler", numpoles=1, RCfact=None):
    """
    Returns list or array of rc_current values which in conjuction with R1 creates a diffusion voltage drop
    """

    if discretization == "euler":
        resistor_current_rc = [0]*len(script_1_current)  # Initialize RC resistor current for error calculation
        for k in range(numpoles, len(script_1_current)):  # start from index 1
            # forward euler like in the model of the battery cell
            resistor_current_rc[k] = (script_1_current[k]*delta_T+resistor_current_rc[k-1]*RC1)/(delta_T+RC1)
    else:  # exact
        # resistor_current_rc = np.zeros((numpoles, len(script_1_current)))  # Initialize RC resistor current for error calculation
        # for k in range(numpoles, len(script_1_current)):
        #     resistor_current_rc[:, k] = np.diag(RCfact) * resistor_current_rc[:, k - 1] + (1 - RCfact) * script_1_current[k - 1]
        # resistor_current_rc = np.transpose(resistor_current_rc)  # Close enough to Octave vrcRaw

        resistor_current_rc = [0]*len(script_1_current)  # Initialize RC resistor current for error calculation
        for k in range(numpoles, len(script_1_current)):
            resistor_current_rc[k] = RCfact * resistor_current_rc[k - 1] + (1 - RCfact) * script_1_current[k - 1]

    return resistor_current_rc


def OCVfromSOCtemp(soc, temp, model):
    """
    Extrapolates OCV vector from the soc vector but based on the previous static OCV/SOC test contained in the model.
    """
    index = np.where(np.array(model.temps) == temp)[0][0]
    function = scipy.interpolate.interp1d(model.soc_vector[index], model.ocv_vector[index], fill_value="extrapolate")
    return function(soc)
