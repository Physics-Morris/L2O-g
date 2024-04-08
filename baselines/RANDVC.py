import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm
import multiprocessing
import os.path
import csv
import copy
import joblib
from torchvision import datasets
import torchvision
from pdb import set_trace as bp
import functools
import time
import argparse

from pennylane import AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer, NesterovMomentumOptimizer, QNGOptimizer, RiemannianGradientOptimizer, RMSPropOptimizer, SPSAOptimizer, RotosolveOptimizer, ShotAdaptiveOptimizer
from pennylane import numpy as pnp

parser = argparse.ArgumentParser(description='RANDVC-baselines')
parser.add_argument('--VC_qubits', type=int, default=7, help='Number of qubits in random VC problem')
parser.add_argument('--VC_layers', type=int, default=8, help='Number of layers in random VC problem')
parser.add_argument('--VC_seed', type=int, default=0, help='Seed use for rand VC problem')
parser.add_argument('--VC_rand_ham', default=False, action='store_true', help='random hamiltonian')
parser.add_argument('--lr', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--itr', type=int, default=500, help='optimzer iteration')
parser.add_argument('--repeat', type=int, default=10, help='repeat exp n times')
parser.add_argument('--opt', type=str, default='gd', help='type of optimizer')


args = parser.parse_args()

qubits = args.VC_qubits
layers = args.VC_layers
VC_seed = args.VC_seed
VC_rand_ham = args.VC_rand_ham
step_size = args.lr
n_times = args.repeat
max_iterations = args.itr

def random_hamiltonian(seed):
    # Generate a random Hamiltonian
    np.random.seed(seed)
    terms = []
    for i in range(qubits):
        coef = np.random.uniform(-1, 1)
        pauli = np.random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
        terms.append(coef * pauli(i))
    return qml.Hamiltonian([t.coeffs[0] for t in terms], [t.ops[0] for t in terms])


dev = qml.device("lightning.qubit", wires=qubits)

np.random.seed(VC_seed)
gate_sequence = [[np.random.choice([qml.RX, qml.RY, qml.RZ]) 
                  for _ in range(qubits)] for _ in range(layers)]
rand_ham = random_hamiltonian(VC_seed)


def ansatz(params):
    # Initial layer of RY(pi/4) gates
    for i in range(qubits):
        qml.RY(np.pi / 4, wires=i)

    # Assuming the total number of layers is known and stored in self.layers
    for l in range(layers):
        for i in range(qubits):
            # Calculate the index for params
            index = l * qubits + i
            # Apply the randomly chosen gate for this layer and qubit
            # Using params[index] to access the parameter
            gate_sequence[l][i](params[index], wires=i)

        # 1D ladder of CZ gates
        for i in range(qubits - 1):
            qml.CZ(wires=[i, i + 1])
    return qml.expval(rand_ham)


@qml.qnode(dev, interface="autograd")
def cost(params):
    return ansatz(params)

if args.opt == 'gd':
    gd_cost = []
    for i in range(n_times):
        pnp.random.seed(i)
        init_params = pnp.random.uniform(low=0, high=2 * np.pi, size=qubits*layers, requires_grad=True)

        opt = qml.GradientDescentOptimizer(step_size)

        params = init_params

        cost_i = []

        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost, params)
            cost_i.append(prev_energy)

            energy = cost(params)
            conv = np.abs(energy - prev_energy)
        gd_cost.append(cost_i)

elif args.opt == 'qngd':
    qngd_cost = []
    opt = qml.QNGOptimizer(step_size, lam=0.01, approx="diag")

    for i in range(n_times):
        pnp.random.seed(i)
        init_params = pnp.random.uniform(low=0, high=2*np.pi, size=qubits*layers, requires_grad=True)
        params = init_params
        prev_energy = cost(params)
        cost_i = []

        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost, params)
            cost_i.append(prev_energy)

            energy = cost(params)
            conv = pnp.abs(energy - prev_energy)
        qngd_cost.append(cost_i)

elif args.opt == 'adam':
    adam_cost = []

    opt = qml.AdamOptimizer(step_size)

    for i in range(n_times):
        pnp.random.seed(i)
        init_params = pnp.random.uniform(low=0, high=2*np.pi, size=qubits*layers, requires_grad=True)
        params = init_params
        prev_energy = cost(params)
        cost_i = []

        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost, params)
            cost_i.append(prev_energy)

            energy = cost(params)
            conv = pnp.abs(energy - prev_energy)
        adam_cost.append(cost_i)

elif args.opt == 'mom':
    mom_cost = []

    opt = qml.MomentumOptimizer(step_size)

    for i in range(n_times):
        pnp.random.seed(i)
        init_params = pnp.random.uniform(low=0, high=2*np.pi, size=qubits*layers, requires_grad=True)
        params = init_params
        prev_energy = cost(params)
        cost_i = []

        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost, params)
            cost_i.append(prev_energy)

            energy = cost(params)
            conv = pnp.abs(energy - prev_energy)
        mom_cost.append(cost_i)

filename = 'RANDVC_' + args.opt + '_q' + str(qubits) + '_l' + str(layers) \
            + '_s' + str(VC_seed) + '_lr' + str(args.lr) + '.npz'
storefile = 'results/randvc_results/' + filename
if args.opt == 'gd':
    np.savez(storefile, gd_cost=np.array(gd_cost))
elif args.opt == 'qngd':
    np.savez(storefile, qngd_cost=np.array(qngd_cost))
elif args.opt == 'adam':
    np.savez(storefile, adam_cost=np.array(adam_cost)) 
elif args.opt == 'mom':
    np.savez(storefile, mom_cost=np.array(mom_cost))