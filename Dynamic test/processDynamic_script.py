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
    index = np.where(np.array(model.temps) == temp)[0]
    function = scipy.interpolate.interp1d(model.soc_vector[index], model.ocv_vector[index])
    return function(soc)


def optfn(theGParam,data,model,theTemp, theTemp_index,doHyst):
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

    :param y:
    :param u:
    :param n: order of model
    :return:
    """
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
    # Compute the R factor
    Y = np.zeros((2, 10))
    U = np.ones((2, 10))
    Rtemp = np.concatenate((U, Y))  # this part is checked and matches the one supplied by .m file
    Rtemp = Rtemp.transpose()
    Rtemp = LA.qr(Rtemp)[0]  # not the same
    Rtemp = np.triu(Rtemp)
    R = Rtemp.transpose()  # Rfactor
    R = R[1:4 * i, 1: 4 * i] # Truncate

    ### STEP 1: Calculate oblique and orthogonal projections ###
    Rf = R[3 * i + 1:4 * i,:]  # Future outputs
    Rp = [R[1:i,:], R[2 * i + 1: 3 * i,:]]  # Past inputs and outputs
    Ru = R[1 * i + 1:2 * i, 1: twoi]  # Future inputs
    # Perpendicular future outputs

    Rfp = [Rf[:, 1: twoi] - (Rf[:, 1:twoi] / Ru)*Ru, Rf[:, twoi + 1: 4 * i]]
    # Perpendicular past inputs and outputs
    Rfp = [Rp[:, 1: twoi] - (Rp[:, 1:twoi] / Ru) * Ru, Rp[:, twoi + 1: 4 * i]]

    # The oblique projection is computed as (6.1) in VODM, page 166.
    # obl / Ufp = Yf / Ufp * pinv(Wp / Ufp) * (Wp / Ufp)
    # The extra projection on Ufp(Uf perpendicular) tends to give better numerical conditioning(see algo on VODM page 131)

    # Funny rank check(SVD takes too long)
    # This check is needed to avoid rank deficiency warnings
    if LA.norm(Rpp[:, 3 * i-2:3 * i], 'fro') < 1e-10:
        Ob = (Rfp * LA.pinv(Rpp.transpose())).transpose() * Rp  # Oblique projection
    else:
        Ob = (Rfp / Rpp) * Rp

    # ------------------------------------------------------------------
    # STEP 2: Compute weighted oblique projection and its SVD
    # Extra projection of Ob on Uf perpendicular
    # ------------------------------------------------------------------
    WOW = [Ob[:, 1: twoi] - (Ob[:, 1:twoi]/ Ru)*Ru, Ob[:, twoi + 1: 4 * i]]
    U, S, _ = LA.svd(WOW)
    ss = np.diag(S)

    # ------------------------------------------------------------------
    # STEP 3: Partitioning U into U1 and U2(the latter is not used)
    # ------------------------------------------------------------------
    U1 = U[:, 1: n]  # Determine U1

    # ------------------------------------------------------------------
    # STEP 4: Determine gam = Gamma(i) and gamm = Gamma(i - 1)
    # ------------------------------------------------------------------
    gam = U1 * np.diag(np.sqrt(ss[1:n]))
    gamm = gam[1:(i - 1),:]
    gam_inv = LA.pinv(gam)  # Pseudo inverse of gam
    gamm_inv = LA.pinv(gamm)  # Pseudo inverse of gamm

    # ------------------------------------------------------------------
    # STEP 5: Determine A matrix(also C, which is not used)
    # ------------------------------------------------------------------
    Rhs = [[gam_inv * R[3 * i + 1:4 * i, 1: 3 * i], zeros(n, 1)], R[i + 1: twoi, 1: 3 * i + 1]]
    Lhs = [[gamm_inv * R[3 * i + 1 + 1:4 * i, 1: 3 * i + 1]], R[3 * i + 1: 3 * i + 1, 1: 3 * i + 1]]
    sol = Lhs / Rhs  # Solve least squares for [A;C]
    A = sol[1:n, 1: n]  # Extract A
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
    rmserr = np.zeros(1, xplots * yplots)

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
        fac = np.exp(-abs(G * np.array(etaik) / (3600 * Q)))
        for k in range(1, len(ik)):  # check if range is same as in matlab
            h[k] = fac[k - 1] * h[k - 1] + (fac[k - 1] - 1) * np.sign(ik[k - 1])
            sik[k] = np.sign(ik[k])
            if abs(ik[k]) < Q / 100:
                sik[k] = sik[k-1]

        # First modeling step: Compute error with model = OCV only
        vest1 = data[ind].OCV
        v_error = np.array(vk) - np.array(vest1)
        numpoles_2 = numpoles

        # Second modeling step: Compute time constants in "A" matrix
        while True:
            A = SISOsubid(-np.diff(v_error), np.diff(etaik), numpoles_2)   # diff works fine
            eigA = LA.eig(A)
            # eigA = eigA(eigA == conj(eigA))  # make sure real # ToDo should be enabled
            # eigA = eigA(eigA > 0 & eigA < 1)  # make sure in range # ToDo should be enabled
            okpoles = len(eigA)
            numpoles_2 = numpoles_2 + 1
            if okpoles >= numpoles_2:
                break
            fprintf('Trying numpoles_2 = %d\n', numpoles_2)

    cost =1
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


