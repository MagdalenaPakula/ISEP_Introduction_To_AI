from pgmpy.models import BayesianModel

from pgmpy.factors.discrete import TabularCPD

# Create the Bayesian network
bayesNet = BayesianModel()

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

# Define conditional probability distributions (CPDs)
cpd_l = TabularCPD("L", 2, [[0.7, 0.3]])
cpd_i = TabularCPD("I", 2, [[0.2, 0.9, 0.3, 0.7],
                            [0.8, 0.1, 0.7, 0.3]],
                   evidence=["L", "S"], evidence_card=[2, 2])
cpd_s = TabularCPD("S", 2, [[0.95, 0.05]])
cpd_n = TabularCPD("N", 2, [[0.9, 0.1]])
cpd_r = TabularCPD("R", 2, [[0.99, 0.01]],
                   evidence=["N"], evidence_card=[2])

# Add CPDs to the network
bayesNet.add_cpds(cpd_l, cpd_i, cpd_s, cpd_n, cpd_r)

# Check if the network structure and CPDs are valid
bayesNet.check_model()

