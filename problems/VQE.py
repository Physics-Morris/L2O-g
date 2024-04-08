import torch
import pennylane as qml
from utils.meta_module import *
import numpy as np


class VQE_Problem:
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        self.hamiltonian = None
        self.qubits = None
        self.hf_state = None

    def setup_problem(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_loss_function(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_optimizee(self):
        raise NotImplementedError("Subclasses should implement this method.")

class VQE_H2_Problem(VQE_Problem):
    def __init__(self, molname="H2", bondlength=0.7):
        dataset = qml.data.load('qchem', molname=molname, bondlength=bondlength)[0]
        device = qml.device("default.qubit", wires=len(dataset.hamiltonian.wires))
        super().__init__(dataset, device)
        self.setup_problem()
        self.define_qnode()

    def setup_problem(self):
        self.hamiltonian, self.qubits = self.dataset.hamiltonian, len(self.dataset.hamiltonian.wires)
        self.hf_state = torch.tensor([1, 1, 0, 0], requires_grad=False)

    def ansatz(self, params, wires=[0, 1, 2, 3]):
        qml.BasisState(self.hf_state, wires=wires)
        for i in wires:
            qml.RZ(params[3 * i], wires=i)
            qml.RY(params[3 * i + 1], wires=i)
            qml.RZ(params[3 * i + 2], wires=i)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])
        return qml.expval(self.hamiltonian)

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