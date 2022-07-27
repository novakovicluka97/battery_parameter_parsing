# serialisation and deserialisation of the objects (class objects, variables, arrays)
from typhoon.api.schematic_editor import SchematicAPI
import pickle
import numpy as np
print("SchematicAPI is imported")
print("Pickle is imported")

#####    self.temps = []
####    self.QParam = []
####    self.etaParam = []
###    self.QParam_static = [] ??? # not to use this parameter
###    self.etaParam_static = [] ???  # not to use this parameter
#    self.RCParam = [] ??
#    self.RParam = [] ??
#####    self.soc_vector = []  # should be multidimensional arrays
#####    self.ocv_vector = []  # should be multidimensional arrays
####    self.GParam = []  # model.GParam = [0] * len(data)
####    self.M0Param = []  # model.M0Param = [0] * len(data)
####    self.MParam = []  # model.MParam = [0] * len(data)
#    self.R0Param = []  # model.R0Param = [0] * len(data)
model = SchematicAPI()
print("Loading model...")
model.load("simple_battery_cell_model.tse")
battery_cell  = model.get_item("Battery Cell", item_type="component")

#def importing_data_obj(battery_object):
pickle_in = open("P14_model.pickle","rb")
battery_object = pickle.load(pickle_in)

#For the model to compile, the length of State of charge vector must be equal to the number of columns in Open circuit voltage vector.
# If the parameter Open circuit voltage vector contains a two dimensional array-like element,
# then the rows must correspond to values inside the Temperatures vector parameter.

#def setting_data_to_model(battery_object_pickle):

#Basic parameters
#setting the soc vector (SOC_vector) [201x3]
model.set_property_value(model.prop(battery_cell, "SOC_vector"),battery_object.soc_vector[0].tolist())

#setting the temperature (T_vector) [3]
model.set_property_value(model.prop(battery_cell, "T_vector"),battery_object.temps)

#setting the ocv vector (OCV) [201x3]
model.set_property_value(model.prop(battery_cell, "OCV"),[battery_object.ocv_vector[0].tolist(),battery_object.ocv_vector[1].tolist(),battery_object.ocv_vector[2].tolist()])
#ocv = f(SOC_vec,T_vector) check the documetation!!!

#setting the R0Param (R0) [48] ??????
model.set_property_value(model.prop(battery_cell, "R0"),battery_object.R0Param)

#setting the etaParam (eta) [3]
model.set_property_value(model.prop(battery_cell, "eta"),battery_object.etaParam)

#setting the QParam (Q_total) [3]
model.set_property_value(model.prop(battery_cell, "nom_Q_combo"),"Total capacity")
model.set_property_value(model.prop(battery_cell, "Q_total"),battery_object.QParam)

##defiusion parameters

if len(battery_object.RCParam) <= 3:
    model.set_property_value(model.prop(battery_cell, "RC_NO"), len(battery_object.RCParam)) #[48] ??
    for i in range(1,len(battery_object.RCParam)):
        model.set_property_value(model.prop(battery_cell, "R" + str(i)), battery_object.RParam[i])
        model.set_property_value(model.prop(battery_cell, "C" + str(i)), battery_object.RCParam[i] / battery_object.RParam[i])
else:
    raise Exception("The length is greater than the maximum model order, the length of RCParam is {}".format(len(battery_object.RCParam)))



#voltage hysteresis tab
model.set_property_value(model.prop(battery_cell, "H_type"),"One state")
model.set_property_value(model.prop(battery_cell, "T_vector_h"), battery_object.temps) #[3] -> this has to be 48,right?
model.set_property_value(model.prop(battery_cell, "M0"), battery_object.M0Param) #[48] ??
model.set_property_value(model.prop(battery_cell, "M"), battery_object.MParam) # [48] ??
model.set_property_value(model.prop(battery_cell, "gamma"), battery_object.GParam) #[48] ??

model.compile()
model.save()
print("Model is saved")
model.close_model()






