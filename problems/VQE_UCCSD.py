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

class VQE_UCCSD_Problem(VQE_Problem):
    def __init__(self, molname="H2", bondlength=0.7, charge=0):
        dataset = qml.data.load('qchem', molname=molname,
                                bondlength=bondlength, basis='STO-3G')[0]
        device = qml.device("default.qubit", wires=len(dataset.hamiltonian.wires))
        super().__init__(dataset, device)
        self.charge = charge
        self.setup_problem()
        self.define_qnode()
        print('# of qubits:', self.qubits, '# of singels', self.singles, '# of doubles', self.doubles)

    def setup_problem(self):
        # print("Dataset content:", self.dataset)
        self.hamiltonian, self.qubits = self.dataset.hamiltonian, len(self.dataset.hamiltonian.wires)
        self.hf_state = self.dataset.hf_state
        symbols = self.dataset.molecule.symbols
        num_electrons = self.calculate_electrons(symbols, self.charge)
        self.singles, self.doubles = qml.qchem.excitations(num_electrons, self.qubits)
        self.s_wires, self.d_wires = qml.qchem.excitations_to_wires(self.singles, self.doubles)

    def calculate_electrons(self, symbols, charge=0):
        electron_count = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
        'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
        'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12,
        'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
        'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 'Cd': 12,
        'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8,
        'Cs': 1, 'Ba': 2,
        'La': 3, 'Ce': 4, 'Pr': 5, 'Nd': 6, 'Pm': 7, 'Sm': 8, 'Eu': 9, 'Gd': 10, 'Tb': 11, 'Dy': 12,
        'Ho': 13, 'Er': 14, 'Tm': 15, 'Yb': 16, 'Lu': 17,
        'Th': 4, 'Pa': 5, 'U': 6, 'Np': 7, 'Pu': 8, 'Am': 9, 'Cm': 10,
        }
        num_electrons = sum(electron_count.get(symbol, 0) for symbol in symbols)
        num_electrons -= charge
        return num_electrons

    def ansatz(self, params, wires):
        qml.UCCSD(params, wires=wires, s_wires=self.s_wires,
                  d_wires=self.d_wires, init_state=self.hf_state)
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
        return len(self.singles) + len(self.doubles)
