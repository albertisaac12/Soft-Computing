import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Input variables
i_temparature = ctrl.Antecedent(np.arange(0,31,1),"i_temparature")
i_humidity = ctrl.Antecedent(np.arange(0,101,1),"i_humidity")
o_temperature = ctrl.Antecedent(np.arange(0,51,1),"o_temperature")
co2_concentration = ctrl.Antecedent(np.arange(0,11,1),"co2_concentration")

# Output variable
fan_air_intensity = ctrl.Consequent(np.arange(0,101,1),"fan_air_intensity")

# Membership functions
i_temparature["cold"] = fuzz.trimf(i_temparature.universe,[0,0,15])
i_temparature["warm"] = fuzz.trimf(i_temparature.universe,[0,20,25])
i_temparature["hot"] = fuzz.trimf(i_temparature.universe,[25,30,30])

i_humidity['low'] = fuzz.trimf(i_humidity.universe, [0, 0, 50])
i_humidity['medium'] = fuzz.trimf(i_humidity.universe, [0, 50, 100])
i_humidity['high'] = fuzz.trimf(i_humidity.universe, [50, 100, 100])

o_temperature["cold"] = fuzz.trimf(o_temperature.universe,[0,0,18])
o_temperature["warm"] = fuzz.trimf(o_temperature.universe,[18,25,30])
o_temperature["hot"] = fuzz.trimf(o_temperature.universe,[30,50,50])

co2_concentration["low"] = fuzz.trimf(co2_concentration.universe,[0,0,2])
co2_concentration["medium"] = fuzz.trimf(co2_concentration.universe,[2,5,7])
co2_concentration["high"] = fuzz.trimf(co2_concentration.universe,[7,10,10])

fan_air_intensity["low"] = fuzz.trimf(fan_air_intensity.universe,[0,25,25])
fan_air_intensity["medium"] = fuzz.trimf(fan_air_intensity.universe,[0,25,50])
fan_air_intensity["high"] = fuzz.trimf(fan_air_intensity.universe,[25,50,50])

fan_air_intensity.defuzzify_method = "MOM"

# Rules
rule1 = ctrl.Rule((i_temparature["hot"] | co2_concentration["high"]) & (i_humidity["low"]),fan_air_intensity["high"])
rule2 = ctrl.Rule((i_temparature["warm"] & i_humidity["medium"]) | (co2_concentration["medium"]),fan_air_intensity["medium"])
rule3 = ctrl.Rule((i_temparature["cold"] & i_humidity["high"]) | (o_temperature["cold"]),fan_air_intensity["low"])
rule4 = ctrl.Rule((i_temparature["hot"] & i_humidity["medium"]) | (o_temperature["warm"]),fan_air_intensity["high"])
rule5 = ctrl.Rule((i_humidity["high"] | co2_concentration["high"]) & (i_temparature["warm"]),fan_air_intensity["high"])
rule6 = ctrl.Rule((i_temparature["cold"] | o_temperature["cold"]) & (co2_concentration["low"]), fan_air_intensity["low"])
rule7 = ctrl.Rule((i_humidity["low"] | i_humidity["medium"]) & (i_temparature["warm"]),fan_air_intensity["medium"])
rule8 = ctrl.Rule((co2_concentration["medium"] | co2_concentration["high"]) & (i_temparature["hot"]),fan_air_intensity["high"])
rule9 = ctrl.Rule((i_temparature["cold"] & i_humidity["low"]) | (o_temperature["warm"]),fan_air_intensity["medium"])
rule10 = ctrl.Rule((i_temparature["warm"] | i_temparature["hot"]) & (i_humidity["high"]), fan_air_intensity["medium"])
rule11 = ctrl.Rule((o_temperature["hot"] | co2_concentration["high"]) & (i_temparature["warm"]),fan_air_intensity["high"])
rule12 = ctrl.Rule((i_temparature["cold"] & i_humidity["medium"]) | (o_temperature["cold"] & co2_concentration["medium"]),fan_air_intensity["low"])

# System
system = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6,
    rule7, rule8, rule9, rule10, rule11, rule12
])

simulation = ctrl.ControlSystemSimulation(system)

# FIXED: matches the variable name i_temparature
simulation.input['i_temparature'] = 25
simulation.input['i_humidity'] = 60
simulation.input['o_temperature'] = 30
simulation.input['co2_concentration'] = 2

simulation.compute()

print("Fan Intensity:", simulation.output['fan_air_intensity'])
