import torch
import pennylane as qml
from utils.meta_module import *
import numpy as np
import networkx as nx

def generate_random_graph(n_nodes, edge_prob, seed=None):
    return nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)

def maxcut_hamiltonian(graph):
    coeffs = []
    terms = []
    for i, j in graph.edges():
        coeffs.append(1)  # Each edge contributes a term in the Hamiltonian
        terms.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, terms)

class QAOA:
    def __init__(self, device):
        self.device = device
        self.gate_set = [qml.RX, qml.RY, qml.RZ]

    def setup_problem(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_loss_function(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_optimizee(self):
        raise NotImplementedError("Subclasses should implement this method.")

class QAOA_problem(QAOA):
    def __init__(self, n_nodes=3, p_layers=1, edge_prob=0.5, seed=0, qaoa_type='MaxCut'):
        self.n_nodes = n_nodes
        self.p_layers = p_layers
        self.edge_prob = edge_prob
        self.seed = seed
        device = qml.device("lightning.qubit", wires=self.n_nodes)
        super().__init__(device)
        self.setup_problem(qaoa_type)
        self.define_qnode()

    def setup_problem(self, qaoa_type='MaxCut'):
        np.random.seed(self.seed)
        if qaoa_type == 'MaxCut':
            self.graph = generate_random_graph(self.n_nodes, self.edge_prob, seed=self.seed)
            self.H_C = maxcut_hamiltonian(self.graph)
        else: print('Unrecognize qaoa type', qaoa_type)

    def ansatz(self, params):
        gamma, beta = params[:self.p_layers], params[self.p_layers:]
        for i in range(self.n_nodes):
            qml.Hadamard(wires=i)
        
        for layer in range(self.p_layers):
            qml.templates.ApproxTimeEvolution(self.H_C, gamma[layer], 1)
            for i in range(self.n_nodes):
                qml.RX(2 * beta[layer], wires=i)
        return qml.expval(self.H_C)


    def define_qnode(self):
        self.qnode = qml.QNode(self.ansatz, self.device, interface="torch")

    def get_metric_fn(self):
        metric_fn = lambda p: qml.metric_tensor(self.qnode, approx="diag")(p)
        return metric_fn

    def get_loss_function(self):
        class Loss_Func:
            def __init__(self, qnode):
                self.qnode = qnode

            def get_loss(self, theta):
                return self.qnode(theta)
        return Loss_Func(self.qnode)