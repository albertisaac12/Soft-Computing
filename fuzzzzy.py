"""
60-70% after midterms
40-30% before midterms

20 - 2 marks each 
5 - 10 marks each
70 marks total

Unit 1 => Soft computing(approximate outcome) and Hard computing(exact outcome)

Compnents of Soft computing => ANN , GA, Swarm Intelligence, Fuzzy Logic (Fuzzy mimics the human decision making process)

Fuzzy Membership vs Probability


"""

# Fuzzy logic => Fuzzy set, Fuzzy rules, Fuzzy inference , aggregation and defuzzification
import matplotlib.pyplot as plt
import numpy as np

fuzzy_set= {
    "Temperature":{
        "low":[0,0,15],
        "medium":[15,15,30],
        "high":[30,30,45]
    },
    "Humidity":{
        "low":[0,0,50],
        "medium":[45,60,80],
        "high":[80,100,100]
    },
    "Fan Speed":{
        "low":[0,0,25],
        "medium":[25,25,50],
        "high":[50,50,100]
    }
}

# Defined fuzzy rules using a structured list of dictionaries
# This is better than strings because it is easier to parse and iterate over programmatically.
fuzzy_rules = [
    {
        "antecedents": {"Temperature": "low", "Humidity": "low"},
        "consequent": {"Fan Speed": "low"}
    },
    {
        "antecedents": {"Temperature": "medium", "Humidity": "medium"},
        "consequent": {"Fan Speed": "medium"}
    },
    {
        "antecedents": {"Temperature": "high", "Humidity": "high"},
        "consequent": {"Fan Speed": "high"}
    },
    {
        "antecedents": {"Temperature": "high", "Humidity": "low"},
        "consequent": {"Fan Speed": "medium"}
    }
]

# Function to calculate membership degree for a given value and membership function

def triangular_membership(x,a,b,c):
    if x <= a or x >= c:
        return 0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def membership_degree(fuzzy_set, **inputs):
    memberships = {}

    for variable, crisp_value in inputs.items():  # e.g., Temperature=20
        memberships[variable] = {}

        for fuzzy_label, params in fuzzy_set[variable].items():  
            a, b, c = params  # unpack the triangular membership parameters
            μ = triangular_membership(crisp_value, a, b, c)
            memberships[variable][fuzzy_label] = μ

    return memberships


def fuzzy_inference(fuzzy_rules, fuzzy_set):
    pass