import scipy.io
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import optimize
fminbnd = scipy.optimize.fminbound
from scipy import interpolate
import matplotlib.pyplot as plt
import generate_battery_cell_data as cell_data
import generate_pickled_cell_model as cell_model
import os
import pandas as pd
import plotly.express as px
import streamlit as st


def get_h_list(script_1_current, G, Q, eta, delta_T=1):
    """
    Returns list of h values which in conjuction with M and M0 params creates a hysteresis voltage drop
    EXPECTS CURRENT THAT IS POSITIVE WHEN IT IS FLOWING OUT OF THE BATTERY!!!
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


def get_rc_current(script_1_current, delta_T=1, RC=None, discretization="euler", numpoles=1, RCfact=None):
    """
    Returns array of rc_current values which in conjunction with R1 creates a diffusion voltage drop
    """

    resistor_current_rc = np.zeros((numpoles, len(script_1_current)))
    if discretization == "euler":
        for numpole in range(numpoles):
            for k in range(1, len(script_1_current)):  # start from index 1
                # forward euler like in the typhoon model of the battery cell component
                resistor_current_rc[numpole, k] = \
                    (script_1_current[k]*delta_T+resistor_current_rc[numpole, k-1]*RC[numpole])/(delta_T+RC[numpole])
    else:  # exact
        # original implementation
        # resistor_current_rc = np.zeros((numpoles, len(script_1_current)))  # Initialize RC resistor current for error calculation
        # for k in range(numpoles, len(script_1_current)):
        #     resistor_current_rc[:, k] = np.diag(RCfact) * resistor_current_rc[:, k - 1] + (1 - RCfact) * script_1_current[k - 1]
        # resistor_current_rc = np.transpose(resistor_current_rc)  # Close enough to Octave vrcRaw
        for numpole in range(numpoles):
            for k in range(1, len(script_1_current)):  # start from index 1
                # forward euler like in the typhoon model of the battery cell component
                resistor_current_rc[numpoles, k] = RCfact[numpole] * resistor_current_rc[k - 1] + \
                                                   (1 - RCfact[numpole]) * script_1_current[k - 1]

    return resistor_current_rc


def save_and_show_data(model, dynamic_data, numpoles):
    """
    Saves all the data in the excel spreadsheet and SISOSubid.mat file
    """
    excel_data, row_index = [], []  # for result compilation

    # Printing the output cell parameters
    print(f"\nMinimization algorithm used: Fminbdn for Gamma and {cell_model.minimization} for R0, R1, RC")
    print(f"Printout of model params:\n")
    print(f"{model.temps=}  Relative error: {error_func(model.temps, 'temps')}")
    print(f"{model.etaParam_static=}  Relative error: {error_func(model.etaParam_static, 'etaParam_static')}")
    print(f"{model.etaParam=}  Relative error: {error_func(model.etaParam, 'etaParam')}")
    print(f"{model.QParam_static=}  Relative error: {error_func(model.QParam_static, 'QParam_static')}")
    print(f"{model.QParam=}  Relative error: {error_func(model.QParam, 'QParam')}")
    print(f"{model.R0Param=}  Relative error: {error_func(model.R0Param, 'R0Param')}")
    print(f"{model.R1Param=}  Relative error: {error_func(model.R1Param, 'R1Param')}")
    print(f"{model.RC1Param=}  Relative error: {error_func(model.RC1Param, 'RC1Param')}")
    print(f"{model.M0Param=}  Relative error: {error_func(model.M0Param, 'M0Param')}")
    print(f"{model.MParam=}  Relative error: {error_func(model.MParam, 'MParam')}")
    print(f"{model.GParam=}  Relative error: {error_func(model.GParam, 'GParam')}")
    print(f"cell_model.ocv_vector at 25 degrees RMS error: {error_func(model.ocv_vector[1], 'OCV')}")
    try:
        plot_func([model.soc_vector[1], cell_data.SOC_default],
                  [model.ocv_vector[1], cell_data.OCV_default[1]],
                  [f"OCV vs SOC graph (Colorado, octave vs {cell_model.data_origin}) for 25 celsius",
                   f"OCV vs SOC graph (Colorado, octave vs {cell_model.data_origin}) for 25 celsius"],
                  flag_show=False)
        plot_func([cell_data.SOC_default],
                  [np.array(cell_data.OCV_default[0]) - np.array(model.ocv_vector[0])],
                  ['T5 RMS error in OCV [V] as a function of SOC'],
                  flag_show=False)
        plot_func([cell_data.SOC_default],
                  [np.array(cell_data.OCV_default[1]) - np.array(model.ocv_vector[1])],
                  ['T25 RMS error in OCV [V] as a function of SOC'],
                  flag_show=False)
        plot_func([cell_data.SOC_default],
                  [np.array(cell_data.OCV_default[2]) - np.array(model.ocv_vector[2])],
                  ['T45 RMS error in OCV [V] as a function of SOC'],
                  flag_show=False)
    except:
        print(f"Unable to plot {cell_model.data_origin}")
        pass

    for k in range(len(model.temps)):
        # Second step: Compute OCV for "discharge portion" of test, based on static test data
        corrected_current = [0] * len(dynamic_data[k].script1.current)
        for index, current in enumerate(dynamic_data[k].script1.current):
            if current > 0:  # if current is flowing into the battery and charging it, apply a coefficient
                corrected_current[index] = current * model.etaParam[k]
            else:
                corrected_current[index] = current

        V_resistor = model.R0Param[k] * np.array(dynamic_data[k].script1.current)
        V_resistor_reference = np.array(dynamic_data[k].script1.current) * np.array(
            dynamic_data[k].script1.internal_resistance)
        rc_current = get_rc_current(dynamic_data[k].script1.current, delta_T=1, RC=[model.RC1Param[k], model.RC2Param[k]],
                                    discretization="euler", numpoles=2)

        V_diff = np.zeros((len(rc_current[0])))
        if numpoles == 3:
            V_diff = rc_current[0] * model.R1Param[k] + rc_current[1] * model.R2Param[k] + rc_current[2] * model.R3Param[k]
        elif numpoles == 2:
            V_diff = rc_current[0] * model.R1Param[k] + rc_current[1] * model.R2Param[k]
        elif numpoles == 1:
            V_diff = rc_current[0] * model.R1Param[k]

        h = get_h_list(-np.array(dynamic_data[k].script1.current),
                       model.GParam[k], model.QParam[k], eta=model.etaParam[k], delta_T=1)
        V_h = np.array(h) * model.MParam[k] + np.sign(dynamic_data[k].script1.current) * model.M0Param[k]
        err = np.array(dynamic_data[k].script1.voltage) - (np.array(dynamic_data[k].OCV) + V_resistor + V_diff + V_h)

        excel_data.append(np.array(dynamic_data[k].script1.voltage))  # reference dynamic voltage data
        row_index.append(f'{model.temps[k]}C ref dynamic voltage')
        excel_data.append(np.array(dynamic_data[k].OCV) + V_resistor + V_diff + V_h)  # generated dynamic voltage data
        row_index.append(f'{model.temps[k]}C gen dynamic voltage')
        excel_data.append([1000 * np.sqrt(np.mean(err ** 2))] * len(err))
        row_index.append(f'{model.temps[k]}C Total RMS')

        excel_data.append(dynamic_data[k].script1.OCV_real)  # reference OCV data
        row_index.append(f'{model.temps[k]}C ref OCV')
        excel_data.append(dynamic_data[k].OCV)  # generated OCV data
        row_index.append(f'{model.temps[k]}C gen OCV')
        excel_data.append([1000 * np.sqrt(
            np.mean((np.array(dynamic_data[k].script1.OCV_real) - np.array(dynamic_data[k].OCV)) ** 2))] * len(err))
        row_index.append(f'{model.temps[k]}C OCV RMS')

        excel_data.append(V_resistor_reference)  # reference internal resistance voltage drop
        row_index.append(f'{model.temps[k]}C ref I*R0')
        excel_data.append(V_resistor)  # generated internal resistance voltage drop
        row_index.append(f'{model.temps[k]}C gen I*R0')
        excel_data.append([1000 * np.sqrt(np.mean((V_resistor_reference - V_resistor) ** 2))] * len(err))
        row_index.append(f'{model.temps[k]}C I*R0 RMS')

        excel_data.append(-np.array(dynamic_data[k].script1.voltage_diffusion))  # reference diffusion voltage drop
        row_index.append(f'{model.temps[k]}C ref V_diff')
        excel_data.append(V_diff)  # generated diffusion voltage drop
        row_index.append(f'{model.temps[k]}C gen V_diff')
        excel_data.append(
            [1000 * np.sqrt(np.mean((np.array(dynamic_data[k].script1.voltage_diffusion) + V_diff) ** 2))] * len(err))
        row_index.append(f'{model.temps[k]}C V_diff RMS')

        excel_data.append(dynamic_data[k].script1.voltage_hysteresis)  # reference hysteresis voltage drop
        row_index.append(f'{model.temps[k]}C ref V_hyst')
        excel_data.append(V_h)  # generated hysteresis voltage drop
        row_index.append(f'{model.temps[k]}C gen V_hyst')
        excel_data.append(
            [1000 * np.sqrt(np.mean((np.array(dynamic_data[k].script1.voltage_hysteresis) - V_h) ** 2))] * len(err))
        row_index.append(f'{model.temps[k]}C V_hyst RMS')

        # Data needed to determine RC circuits is saved for external processing as well as excel data for visualizing
        SISOSubid_data = {'verr': np.array(dynamic_data[k].script1.voltage) - np.array(dynamic_data[k].OCV),
                          'curr_corrected': corrected_current,
                          'curr': dynamic_data[k].script1.current}
        scipy.io.savemat("SUB_ID_" + str(model.temps[k]) + ".mat", SISOSubid_data)

    #
    df = pd.DataFrame(data=excel_data,
                      index=row_index)
    df = df.T
    # df.to_excel(cell_model.data_origin + ".xlsx")
    # print("Excel data saved as ", cell_model.data_origin + ".xlsx")
    # os.system("start EXCEL.EXE " + cell_model.data_origin + ".xlsx")

    st.set_page_config(page_title="Battery cell parametrization", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.dataframe(df)

    df_selection = df

    # Main page 1
    st.title(":bar_chart: Results Dashboard")
    st.markdown("##")

    total_rms_25C = round(df_selection['25C Total RMS'].mean(), 2)
    total_rms_5C = round(df_selection['5C Total RMS'].mean(), 2)
    total_rms_45C = round(df_selection['45C Total RMS'].mean(), 2)

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total RMS error for the dynamic script for 5 degrees C:")
        st.subheader(f"Voltage RMS = {total_rms_5C} mV")
    with middle_column:
        st.subheader("Total RMS error for the dynamic script for 25 degrees C:")
        st.subheader(f"Voltage RMS = {total_rms_25C} mV")
    with right_column:
        st.subheader("Total RMS error for the dynamic script for 45 degrees C:")
        st.subheader(f"Voltage RMS = {total_rms_45C} mV")

    st.markdown("""---""")

    # Plot
    st.title(":bar_chart: Graph of dynamic script voltage for 25C (estimation vs original)")
    st.line_chart(data=df, y=['25C ref dynamic voltage', '25C gen dynamic voltage'], width=0, height=500,
                  use_container_width=True)

    # Main page 2
    st.title(f":chart_with_upwards_trend: Minimization algorithm used: Fminbdn for Gamma and {cell_model.minimization} for R0, R1, RC")
    st.markdown("##")

    column_0_Q = pd.DataFrame(data=[np.array(model.QParam),
                                    np.array(cell_data.QParam)],
                              index=["QParam", "QParam original"],
                              columns=["5", "25", "45"])
    column_1_eta = pd.DataFrame(data=[np.array(model.etaParam),
                                    np.array(cell_data.etaParam)],
                              index=["etaParam", "etaParam original"],
                              columns=["5", "25", "45"])
    column_2_R0 = pd.DataFrame(data=[np.array(model.R0Param),
                                    np.array(cell_data.R0Param)],
                              index=["R0Param", "R0Param original"],
                              columns=["5", "25", "45"])
    column_3_R1 = pd.DataFrame(data=[np.array(model.R1Param),
                                     np.array(cell_data.R1Param)],
                               index=["R1Param", "R1Param original"],
                               columns=["5", "25", "45"])
    column_4_RC1 = pd.DataFrame(data=[np.array(model.RC1Param),
                                     np.array(cell_data.RC1param)],
                               index=["RC1Param", "RC1Param original"],
                               columns=["5", "25", "45"])
    column_5_G = pd.DataFrame(data=[np.array(model.GParam),
                                    np.array(cell_data.GParam)],
                              index=["GParam", "GParam original"],
                              columns=["5", "25", "45"])
    column_6_M = pd.DataFrame(data=[np.array(model.MParam),
                                    np.array(cell_data.MParam)],
                              index=["MParam", "MParam original"],
                              columns=["5", "25", "45"])
    column_7_M0 = pd.DataFrame(data=[np.array(model.M0Param),
                                    np.array(cell_data.M0Param)],
                              index=["M0Param", "M0Param original"],
                              columns=["5", "25", "45"])

    if cell_data.doHyst == 1:
        column_0, column_1, column_2, column_3, column_4, column_5, column_6, column_7 = st.columns(8)
        with column_0:
            fig, ax = plt.subplots()
            column_0_Q.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_1:
            fig, ax = plt.subplots()
            column_1_eta.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_2:
            fig, ax = plt.subplots()
            column_2_R0.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_3:
            fig, ax = plt.subplots()
            column_3_R1.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_4:
            fig, ax = plt.subplots()
            column_4_RC1.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_5:
            fig, ax = plt.subplots()
            column_5_G.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_6:
            fig, ax = plt.subplots()
            column_6_M.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_7:
            fig, ax = plt.subplots()
            column_7_M0.T.plot.bar(ax=ax)
            st.pyplot(fig)
    else:
        column_0, column_1, column_2, column_3, column_4 = st.columns(5)
        with column_0:
            fig, ax = plt.subplots()
            column_0_Q.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_1:
            fig, ax = plt.subplots()
            column_1_eta.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_2:
            fig, ax = plt.subplots()
            column_2_R0.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_3:
            fig, ax = plt.subplots()
            column_3_R1.T.plot.bar(ax=ax)
            st.pyplot(fig)
        with column_4:
            fig, ax = plt.subplots()
            column_4_RC1.T.plot.bar(ax=ax)
            st.pyplot(fig)

    # Main page 3
    st.markdown("##")

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader(f"Printout of model params:")
        st.subheader("temps=          " + f"{model.temps}")
        st.subheader("etaParam_static=" + f"{model.etaParam_static}")
        st.subheader("etaParam=       " + f"{model.etaParam}")
        st.subheader("QParam_static=  " + f"{model.QParam_static}")
        st.subheader("QParam=         " + f"{model.QParam}")
        st.subheader("R0Param=        " + f"{model.R0Param}")
        st.subheader("R1Param=         " + f"{model.R1Param}")
        st.subheader("RC1Param=        " + f"{model.RC1Param}")
        st.subheader("M0Param=        " + f"{model.M0Param}")
        st.subheader("MParam=         " + f"{model.MParam}")
        st.subheader("GParam=         " + f"{model.GParam}")

    with right_column:
        st.subheader(f"Relative errors:")
        st.subheader(f"{error_func(model.temps, 'temps')}")
        st.subheader(f"{error_func(model.etaParam_static, 'etaParam_static')}")
        st.subheader(f"{error_func(model.etaParam, 'etaParam')}")
        st.subheader(f"{error_func(model.QParam_static, 'QParam_static')}")
        st.subheader(f"{error_func(model.QParam, 'QParam')}")
        st.subheader(f"{error_func(model.R0Param, 'R0Param')}")
        st.subheader(f"{error_func(model.R1Param, 'R1Param')}")
        st.subheader(f"{error_func(model.RC1Param, 'RC1Param')}")
        st.subheader(f"{error_func(model.M0Param, 'M0Param')}")
        st.subheader(f"{error_func(model.MParam, 'MParam')}")
        st.subheader(f"{error_func(model.GParam, 'GParam')}")

    st.markdown("""---""")


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
            elif param_name == 'R1Param':
                error.append(round((cell_data.R1Param[i] - model_param[i]) / cell_data.R1Param[i], 2))
            elif param_name == 'RC1Param':
                error.append(round((cell_data.RC1param[i] - model_param[i]) / cell_data.RC1param[i], 2))
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
        self.soc_vector = []  # should be multidimensional arrays
        self.ocv_vector = []  # should be multidimensional arrays
        self.GParam = []  # model.GParam = [0] * len(data)
        self.M0Param = []  # model.M0Param = [0] * len(data)
        self.MParam = []  # model.MParam = [0] * len(data)
        self.R0Param = []  # model.R0Param = [0] * len(data)
        self.RC1Param = []  # model.RC1Param = [0] * len(data)
        self.R1Param = []  # model.R1Param = [0] * len(data)
        self.C1Param = []  # model.R1Param = [0] * len(data)
        self.RC2Param = []  # model.RC2Param = [0] * len(data)
        self.R2Param = []  # model.R2Param = [0] * len(data)
        self.C2Param = []  # model.R2Param = [0] * len(data)
        self.RC3Param = []  # model.RC3Param = [0] * len(data)
        self.R3Param = []  # model.R3Param = [0] * len(data)
        self.C3Param = []  # model.R3Param = [0] * len(data)


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
                if temp == 6:  # todo change this (remains from Boulder data)
                    self.static_data.append(self.StaticData(MAT_data['OCVData_' + '05'], temp))
                else:
                    self.static_data.append(self.StaticData(MAT_data['OCVData_' + str(temp)], temp))
        except ValueError:
            print('ValueError for static data. Check the format of the provided data')

        try:
            for temp in temp_dyn:
                if temp == 6:  # todo change this (remains from Boulder data)
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
            # time, temp, voltage, current, chgAh, disAh, OCV, Vhyst, Vdiffusion
            def __init__(self, script):
                self.time = script[0][0][0][0]
                self.temp = script[0][0][1][0]
                self.voltage = script[0][0][2][0]
                self.current = script[0][0][3][0]
                self.chgAh = script[0][0][4][0]
                self.disAh = script[0][0][5][0]
                self.OCV_real = script[0][0][6][0]
                self.voltage_hysteresis = script[0][0][7][0]
                self.voltage_diffusion = script[0][0][8][0]
                self.internal_resistance = script[0][0][9][0]

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
    def __init__(self, time, temperature, voltage, current, chgAh, disAh, OCV=None,
                 voltage_diffusion=None, voltage_hysteresis=None, internal_resistance=None):
        if OCV is None:
            OCV = [404]*len(voltage)
        if voltage_diffusion is None:
            voltage_diffusion = [404]*len(voltage)
        if voltage_hysteresis is None:
            voltage_hysteresis = [404]*len(voltage)
        if internal_resistance is None:
            internal_resistance = [404]*len(voltage)
        self.time = time
        self.temperature = temperature
        self.voltage = voltage
        self.current = current
        self.chgAh = chgAh
        self.disAh = disAh
        self.OCV = OCV
        self.voltage_hysteresis = voltage_hysteresis
        self.voltage_diffusion = voltage_diffusion
        self.internal_resistance = internal_resistance
