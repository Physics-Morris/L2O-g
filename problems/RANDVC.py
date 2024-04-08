import torch
import pennylane as qml
from utils.meta_module import *
import numpy as np


class RANDVC:
    def __init__(self, device):
        self.device = device
        self.gate_set = [qml.RX, qml.RY, qml.RZ]

    def setup_problem(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_loss_function(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_optimizee(self):
        raise NotImplementedError("Subclasses should implement this method.")

class RANDVC_problem(RANDVC):
    def __init__(self, qubits=4, layers=1, seed=0, rand_ham=False):
        self.qubits = qubits
        self.layers = layers
        self.seed = seed
        self.rand_ham = rand_ham
        device = qml.device("lightning.qubit", wires=qubits)
        super().__init__(device)
        self.setup_problem()
        self.define_qnode()

    def setup_problem(self):
        np.random.seed(self.seed)
        self.gate_sequence = [[np.random.choice([qml.RX, qml.RY, qml.RZ]) 
                               for _ in range(self.qubits)] for _ in range(self.layers)]
        if self.rand_ham:
            self.hamiltonian = self.random_hamiltonian()
        else:
            self.hamiltonian = qml.PauliZ(0) @ qml.PauliZ(1)

    def random_hamiltonian(self):
        # Generate a random Hamiltonian
        np.random.seed(self.seed)
        terms = []
        for i in range(self.qubits):
            coef = np.random.uniform(-1, 1)
            pauli = np.random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
            terms.append(coef * pauli(i))
        return qml.Hamiltonian([t.coeffs[0] for t in terms], [t.ops[0] for t in terms])

    def ansatz(self, params):
        # Initial layer of RY(pi/4) gates
        for i in range(self.qubits):
            qml.RY(np.pi / 4, wires=i)
        
        for l in range(self.layers):
            for i in range(self.qubits):
                index = l * self.qubits + i
                self.gate_sequence[l][i](params[index], wires=i)
            
            # 1D ladder of CZ gates
            for i in range(self.qubits - 1):
                qml.CZ(wires=[i, i + 1])
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