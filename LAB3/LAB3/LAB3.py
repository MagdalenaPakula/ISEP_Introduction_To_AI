from statistics import variance
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD 
from pgmpy.inference import VariableElimination 
import numpy as np 

# Create the Bayesian network
bayesNet = BayesianNetwork()

# Add nodes
bayesNet.add_node("L")
bayesNet.add_node("I")
bayesNet.add_node("S")
bayesNet.add_node("N")
bayesNet.add_node("R")

# Add edges
bayesNet.add_edge("L", "I")
bayesNet.add_edge("S", "I")
bayesNet.add_edge("N", "L")
bayesNet.add_edge("N", "I")
bayesNet.add_edge("N", "S")
bayesNet.add_edge("N", "R")
    
# Defining conditional probability distributions (CPDs)
cpd_L = TabularCPD(variable="L", variable_card=2, values=[[0.92], [0.08]])
cpd_I = TabularCPD(variable="I", variable_card=2, values=[[0.79], [0.21]])
cpd_S = TabularCPD(variable="S", variable_card=2, values=[[0.88], [0.12]])
cpd_N = TabularCPD(variable="N", variable_card=2,
                   values=[[0.97, 0.83, 0.92, 0.78, 0.27, 0.21, 0.12, 0.08],
                           [0.03, 0.17, 0.08, 0.22, 0.73, 0.79, 0.88, 0.92]],
                   evidence=["L", "S", "I"], evidence_card=[2, 2, 2])
cpd_R = TabularCPD(variable="R", variable_card=2,
                   values=[[0.95, 0.84, 0.92, 0.62], [0.05, 0.16, 0.08, 0.38]],
                   evidence=["N", "S"], evidence_card=[2, 2])

# Add CPDs to the network   
bayesNet.add_cpds(cpd_L, cpd_I, cpd_S, cpd_N, cpd_R)

# Check if the network structure and CPDs are valid
#print(bayesNet.check_model())

#Solver
solver=VariableElimination(bayesNet)

result_N = solver.query(variables="N")
print(result_N)