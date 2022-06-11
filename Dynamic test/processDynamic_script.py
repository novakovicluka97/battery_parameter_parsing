from battery_cell_data import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import optimize


fminbnd = scipy.optimize.fminbound
variable = 1


def OCVfromSOCtemp(soc,temp,model):
    """
    Extrapolates OCV vector from the soc vector but based on the previous static OCV/SOC test. Test results are contained in the model.

    """
    index = np.where(np.array(model.temps) == temp)[0][0]
    function = scipy.interpolate.interp1d(model.soc_vector[index], model.ocv_vector[index])
    return function(soc)


def optfn(theGParam, data, model, theTemp, theTemp_index, doHyst):
    """

    """
    global bestcost
    model.GParam[theTemp_index] = abs(theGParam)
    [cost, model] = minfn(data, model, theTemp, doHyst)
    if cost < bestcost:
        bestcost = cost
        print("The model created for this value of gamma is the best ESC model yet! cost = ", cost)
    return cost


def SISOsubid(y, u, n):
    """
    Etirely debugged
    :param y:
    :param u:
    :param n: order of model
    :return:
    """
    # inputs looks very precise (u seems completely same as in Octave)
    ny = len(y)
    y = np.array(y)  # does this even work? Recheck this if some latter multiplication doesnt work
    # y = y.transpose()  # turn y into row vector

    nu = len(u)
    u = np.array(u)  # does this even work? Recheck this if some latter multiplication doesnt work
    # u = u.transpose()  # turn u into row vector

    i = 2 * n  # two times the number of poles
    twoi = 4 * n
    if nu != ny:
        print("EXCEPTION, nu and ny must be the same size")
    if (ny - twoi + 1) < twoi:
        print("EXCEPTION, not enough data points")

    # Determine the number of columns in the Hankel matrices
    j = ny - twoi + 1  # adds up to 0?

    # Make Hankel matrices Y and U
    Y = np.zeros((twoi, j))
    U = np.zeros((twoi, j))
    for k in range(twoi):
        Y[k, :] =y[k : k + j]
        U[k, :] =u[k : k + j]
    # U looks the same and Y looks 99% same
    # Compute the R factor
    Rtemp = np.concatenate((U, Y))  # this part is checked and matches the one supplied by .m file
    Rtemp = Rtemp.transpose()
    R = LA.qr(Rtemp)[1].transpose()
    # This looks a little cleaner than in Octave but that is because functions work differently
    # Still, the end result for R looks similar to what we get in Octave

    ### STEP 1: Calculate oblique and orthogonal projections ###
    Rf = R[3 * i : 4 * i,:]  # Future outputs
    Rp = np.append(R[0:i,:], R[2 * i: 3 * i,:], 0)  # Past inputs and outputs
    Ru = R[1 * i:2 * i, 0: twoi]  # Future inputs
    # Perpendicular future outputs

    Rfp_var = np.dot(Rf[:, 0:twoi], np.linalg.pinv(Ru))
    Rfp_var1 = np.dot(Rfp_var, Ru)
    Rfp = np.append(Rf[:, 0: twoi] - Rfp_var1, Rf[:, twoi : 4 * i], 1)
    # Perpendicular past inputs and outputs
    Rpp_var = np.dot(Rp[:, 0:twoi], np.linalg.pinv(Ru))
    Rpp_var1 = np.dot(Rpp_var, Ru)
    Rpp = np.append(Rp[:, 0: twoi] - Rpp_var1, Rp[:, twoi: 4 * i], 1)

    # The oblique projection is computed as (6.1) in VODM, page 166.
    # obl / Ufp = Yf / Ufp * pinv(Wp / Ufp) * (Wp / Ufp)
    # The extra projection on Ufp(Uf perpendicular) tends to give better numerical conditioning(see algo on VODM page 131)

    # Funny rank check(SVD takes too long)
    # This check is needed to avoid rank deficiency warnings
    if LA.norm(Rpp[:, 3 * i-2:3 * i], 'fro') < 1e-10:
        Ob = (Rfp * LA.pinv(Rpp.transpose())).transpose() * Rp  # Oblique projection
    else:
        # Ob = (Rfp / Rpp) * Rp
        Ob_var = np.dot(Rfp, np.linalg.pinv(Rpp))
        Ob = np.dot(Ob_var, Rp)

    # ------------------------------------------------------------------
    # STEP 2: Compute weighted oblique projection and its SVD
    # Extra projection of Ob on Uf perpendicular
    # ------------------------------------------------------------------
    WOW_var = np.dot(Ob[:, 0:twoi], np.linalg.pinv(Ru))
    WOW_var1 = np.dot(WOW_var, Ru)
    WOW = np.append(Ob[:, 0: twoi] - WOW_var1, Ob[:, twoi: 4 * i], 1)
    U, S, _ = LA.svd(WOW)
    ss = np.transpose(S)  # ss = np.diag(S)

    # ------------------------------------------------------------------
    # STEP 3: Partitioning U into U1 and U2(the latter is not used)
    # ------------------------------------------------------------------
    U1 = U[:, 0: n]  # Determine U1

    # ------------------------------------------------------------------
    # STEP 4: Determine gam = Gamma(i) and gamm = Gamma(i - 1)
    # ------------------------------------------------------------------
    gam = U1 * np.diag(np.sqrt(ss[0:n]))
    gamm = gam[0:(i - 1),:]
    gam_inv = LA.pinv(gam)  # Pseudo inverse of gam
    gamm_inv = LA.pinv(gamm)  # Pseudo inverse of gamm

    # ------------------------------------------------------------------
    # STEP 5: Determine A matrix(also C, which is not used)
    # ------------------------------------------------------------------
    var_debug0 = np.dot(gam_inv, R[3 * i:4 * i, 0: 3 * i])
    var_debug1 = np.zeros((n, 1))
    var_debug2 = R[i: twoi, 0: 3 * i + 1]
    Rhs_var = np.append(np.dot(gam_inv, R[3 * i:4 * i, 0: 3 * i]), np.zeros((n, 1)), 1)
    Rhs = np.append(Rhs_var, R[i: twoi, 0: 3 * i + 1], 0)
    Lhs = np.append(np.dot(gamm_inv, R[3 * i + 1:4 * i, 0: 3 * i + 1]), R[3 * i : 3 * i + 1, 0: 3 * i + 1], 0)
    sol = np.dot(Lhs, LA.pinv(Rhs))  # (sol = Lhs / Rhs) Solve least squares for [A;C]
    A = sol[0:n, 0: n]  # Extract A
    return A

def minfn(data, model, theTemp, doHyst):
    """

    :return:
    """
    alltemps = [data[i].temp for i in range(len(data))]
    index_array = np.where(np.array(alltemps) == theTemp)  # fixed at the moment
    ind = index_array[0][0]
    numfiles = len(index_array)  # will be 1 for now

    xplots = np.ceil(np.sqrt(numfiles))
    yplots = np.ceil(numfiles / xplots)
    rmserr = np.zeros(1, xplots * yplots)[0]

    G = abs(model.GParam[ind])      # for 25 degrees
    Q = abs(model.QParam[ind])      # for 25 degrees
    eta = abs(model.etaParam[ind])  # for 25 degrees
    RC = model.RCParam[ind]         # for 25 degrees
    numpoles = 1  # len(RC)         # for 25 degrees

    for thefile in range(numfiles):  # should always be 1 file
        ik = data[ind].script1.current[:]
        vk = data[ind].script1.voltage[:]
        tk = range(len(vk))  # works, all good but is never used again
        etaik = ik  # is this a shallow copy?
        for i in range(len(etaik)):
            if etaik[i] < 0:
                etaik[i] = etaik[i] * eta

        h = [0] * len(ik)
        sik = [0] * len(ik)
        fac = np.exp(-abs(G * np.array(etaik) / (3600 * Q)))  # looks the same as octave
        for k in range(1, len(ik)):  # check if range is same as in matlab
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
        RC = -1 / np.log(RCfact)  # reference code says 2.3844
        # Simulate the R - C filters to find R - C currents
        vrcRaw = np.zeros((numpoles, len(h)))
        for k in range(1, len(ik)):
            vrcRaw[:, k] = np.diag(RCfact) * vrcRaw[:, k - 1] + (1 - RCfact) * etaik[k - 1]
        vrcRaw = np.transpose(vrcRaw)
        # Close enough vrcRaw
        # Todo last checkpoint

        # Third modeling step: Hysteresis parameters
        if doHyst:
            H = np.append(np.array(h), np.array(sik)) # -etaik, -vrcRaw)
            W = LA.lstsq(H, v_error) # W = H\verr;   LEAST SQUARE NON NEGATIVE
            M  = W[0]
            M0 = W[1]
            R0 = W[2]
            Rfact = np.transpose(W[2:])
        else:
            H = [-etaik, -vrcRaw]
            W = H / v_error  # Todo probably something wrong here, revisit later
            M = 0
            M0 = 0
            R0 = W[0]
            Rfact = np.transpose(W[1:])
        ind = np.where(model.temps == data[ind[thefile]].temp, 1)
        model.R0Param[ind] = R0
        model.M0Param[ind] = M0
        model.MParam[ind] = M
        model.RCParam[ind,:] = np.transpose(RC)
        model.RParam[ind,:] = np.transpose(Rfact)

        vest2 = vest1 + M * h + M0 * sik - R0 * etaik - vrcRaw * np.transpose(Rfact)
        verr = vk - vest2

        # Compute RMS error only on data roughly in 5 % to 95 % SOC
        v1 = OCVfromSOCtemp(0.95, data[ind[thefile]].temp, model)
        v2 = OCVfromSOCtemp(0.05, data[ind[thefile]].temp, model)
        N1 = np.where(vk < v1, 1, 'first')
        N2 = np.where(vk < v2, 1, 'first')
        if not N1:
            N1=1
        if not N2:
            N2=len(v_error)
        rmserr[thefile] = np.sqrt(np.mean(v_error[N1:N2]**2))

    cost = sum(rmserr)
    print(f'RMS error for present value of gamma = {cost * 1000} (mV)\n')
    assert not cost, 'Exception: Cost is empty'

    return [cost, model]


def processDynamic(data,model,numpoles,doHyst):
    """

    """
    ##### STEP 1: Compute Q and eta ######
    ind25 = None
    for index, data_for_1_temp in enumerate(data):
        if data_for_1_temp.temp == 25:
            ind25 = index

    if ind25 == None:
        print("There is no default temperature of 25 deg celsius!")
        raise Exception()

    totDisAh = data[ind25].script1.disAh[-1] + data[ind25].script2.disAh[-1] + data[ind25].script3.disAh[
        -1]  # if we have chgAh and disAh as arrays
    totChgAh = data[ind25].script1.chgAh[-1] + data[ind25].script2.chgAh[-1] + data[ind25].script3.chgAh[
        -1]  # if we have chgAh and disAh as arrays
    eta25 = totDisAh / totChgAh
    data[ind25].script1.chgAh[-1] = eta25 * data[ind25].script1.chgAh[-1]  # correct for the coefficients
    data[ind25].script2.chgAh[-1] = eta25 * data[ind25].script2.chgAh[-1]  # correct for the coefficients
    data[ind25].script3.chgAh[-1] = eta25 * data[ind25].script3.chgAh[-1]  # correct for the coefficients
    Q25 = data[ind25].script1.disAh[-1] + data[ind25].script2.disAh[-1] - data[ind25].script1.chgAh[-1] - data[ind25].script2.chgAh[-1]
    data[ind25].Q = Q25
    data[ind25].eta = eta25

    for k in range(len(data)):
        if data[k].temp != 25:  # revisit this later (it was copy pasted without understanding)
            data[k].script2.chgAh = data[k].script2.chgAh * eta25
            data[k].script3.chgAh = data[k].script3.chgAh * eta25
            eta = (data[k].script1.disAh[-1] + data[k].script2.disAh[-1] +
                   data[k].script3.disAh[-1] - data[k].script2.chgAh[-1] -
                   data[k].script3.chgAh[-1]) / data[k].script1.chgAh[-1]

            data[k].script1.chgAh = eta * data[k].script1.chgAh
            Q = data[k].script1.disAh[-1] + data[k].script2.disAh[-1] - data[k].script1.chgAh[-1] - data[k].script2.chgAh[-1]
            data[k].Q = Q
            data[k].eta = eta
        model.QParam.append(data[k].Q)
        model.etaParam.append(data[k].eta)
        model.temps.append(data[k].temp)

    ##### STEP 2: OCV ###### for "discharge portion" of test
    for k in range(len(data)):
        corrected_current = [0]*len(data[k].script1.current)
        for index, current in enumerate(data[k].script1.current):
            if current < 0:  # if current is flowing into the battery and charging it
                corrected_current[index] = current * model.etaParam[k]
            else:
                corrected_current[index] = current
        data[k].Z = np.ones(np.size(corrected_current)) - np.cumsum(corrected_current)/(data[k].Q*3600)  # for "discharge portion" of test
        data[k].OCV = OCVfromSOCtemp(data[k].Z, data[k].temp, model)  # for "discharge portion" of test
        # plt.plot(data[k].script1.time, data[k].Z)

    ##### STEP 3: OPTIMIZE #####
    model.GParam  = [0]*len(data)
    model.M0Param = [0]*len(data)
    model.MParam  = [0]*len(data)
    model.R0Param = [0]*len(data)
    model.RCParam = [[0]*len(data)] * numpoles
    model.RParam  = [[0]*len(data)] * numpoles

    for k in range(len(data)):
        bestcost = np.inf
        print("Processing temperature", data[k].temp)
        if doHyst:
            model.GParam[k] = abs(fminbnd(optfn, 1, 250, args=(data, model, model.temps[k], k, doHyst), xtol=0.1, maxfun=40, disp=3))
        else:  # probably never going to happen
            model.GParam[k] = 0
            theGParam = 0
            optfn(theGParam, data, model, model.temps(theTemp), doHyst)
            [a , model] = minfn(data, model, model.temps(theTemp), doHyst)


