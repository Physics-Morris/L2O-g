import torch
import pennylane as qml
from functools import partial
from utils.meta_module import *
import numpy as np


class VQE_Problem:
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        self.hamiltonian = None
        self.qubits = None
        self.layers = None
        self.hf_state = None

    def setup_problem(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_loss_function(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_optimizee(self):
        raise NotImplementedError("Subclasses should implement this method.")

class VQE_RALE_Problem(VQE_Problem):
    def __init__(self, molname="H2", bondlength=0.7, charge=0, layers=1):
        dataset = qml.data.load('qchem', molname=molname,
                                bondlength=bondlength, basis='STO-3G')[0]
        device = qml.device("default.qubit", wires=len(dataset.hamiltonian.wires))
        super().__init__(dataset, device)
        self.charge = charge
        self.layers = layers
        self.setup_problem()
        self.define_qnode()
        print('# of qubits:', self.qubits)

    def setup_problem(self):
        # print("Dataset content:", self.dataset)
        self.hamiltonian, self.qubits = self.dataset.hamiltonian, len(self.dataset.hamiltonian.wires)
        self.hf_state = self.dataset.hf_state

    def ansatz(self, params, wires):
        func = partial(qml.ParticleConservingU1, init_state=self.hf_state, wires=wires)
        shape = qml.ParticleConservingU1.shape(self.layers, self.qubits)
        params = params.reshape(shape)
        func(params)
        return qml.expval(self.hamiltonian)

    def define_qnode(self):
        self.qnode = qml.QNode(lambda params: self.ansatz(params,
                                                          wires=range(self.qubits)),
                                                          self.device, interface="torch")
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

    def get_dimension(self):
        return (self.qubits-1) * self.layers * 2
