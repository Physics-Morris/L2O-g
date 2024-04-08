import argparse
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import time
import csv
import itertools
from pennylane import AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer, NesterovMomentumOptimizer, QNGOptimizer, RiemannianGradientOptimizer, RMSPropOptimizer, SPSAOptimizer, RotosolveOptimizer, ShotAdaptiveOptimizer

def setup_argparser():
    parser = argparse.ArgumentParser(description="Run QAOA experiments for the MaxCut problem.")
    parser.add_argument('--optimizer', type=str, choices=['GD', 'Momentum', 'Adam', 'Adagrad', 'RMSprop', 'QNGD'], required=True, help='Choice of optimizer')
    parser.add_argument('--n_nodes', type=int, required=True, help='Number of nodes in the graph')
    parser.add_argument('--edge_prob', type=float, required=True, help='Probability of edge creation in the graph')
    parser.add_argument('--lr', type=float, required=True, help='learning rate for optimizer')
    parser.add_argument('--p_layers', type=int, required=True, help='Number of layers in the QAOA circuit')
    parser.add_argument('--repeat', type=int, default=1, help='Number of repetitions for the experiment')
    parser.add_argument('--seed', type=int, default=0, help='random graph seed')
    return parser

def generate_random_graph(n_nodes, edge_prob, seed=None):
    return nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)

def maxcut_hamiltonian(graph):
    coeffs = []
    terms = []
    for i, j in graph.edges():
        coeffs.append(1)  # Each edge contributes a term in the Hamiltonian
        terms.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, terms)

def select_optimizer(name, lr):
    step = lr
    if name == 'GD':
        return GradientDescentOptimizer(stepsize=step)
    elif name == 'Momentum':
        return MomentumOptimizer(stepsize=step)
    elif name == 'Adam':
        return AdamOptimizer(stepsize=step)
    elif name == 'Adagrad':
        return AdagradOptimizer(stepsize=step)
    elif name == 'RMSprop':
        return RMSPropOptimizer(stepsize=step)
    elif name == 'QNGD':
        return QNGOptimizer(stepsize=step, lam=0.01, approx="diag")
    else: print('unreg. optimizer', name)

def qaoa_circuit(gamma, beta, n_nodes, p_layers, H_C):
    for i in range(n_nodes):
        qml.Hadamard(wires=i)
    
    for layer in range(p_layers):
        qml.templates.ApproxTimeEvolution(H_C, gamma[layer], 1)
        for i in range(n_nodes):
            qml.RX(2 * beta[layer], wires=i)

def maxcut_bruteforce(graph):
    nodes = list(graph.nodes())
    best_cut_value = 0
    n = len(nodes)

    # Iterate over all possible sets of nodes (all possible cuts)
    for i in range(1 << (n // 2 + 1)):  # Only up to half, to avoid symmetry
        subset = {nodes[j] for j in range(n) if i & (1 << j)}
        cut_size = sum(1 for u, v in graph.edges() if (u in subset) != (v in subset))
        best_cut_value = max(best_cut_value, cut_size)

    return best_cut_value

def random_partition_cut(graph, num_trials=1000):
    nodes = list(graph.nodes())
    cut_sizes = []
    for _ in range(num_trials):
        np.random.shuffle(nodes)
        subset = set(nodes[:len(nodes)//2])
        cut_size = sum(1 for u, v in graph.edges() if (u in subset and v not in subset) or (u not in subset and v in subset))
        cut_sizes.append(cut_size)
    return np.mean(cut_sizes)
    

parser = setup_argparser()
args = parser.parse_args()

optimizer = args.optimizer
lr = args.lr
n_nodes = args.n_nodes
edge_prob = args.edge_prob
p_layers = args.p_layers
repeat = args.repeat
seed = args.seed

# Create a graph
graph = generate_random_graph(n_nodes, edge_prob, seed)
H_C = maxcut_hamiltonian(graph)
# Initialize device
dev = qml.device('lightning.qubit', wires=n_nodes)

# Set optimizer
opt = select_optimizer(optimizer, lr)

# Prepare file for logging
filename = f"results/maxcut_results/{optimizer}_lr{lr}_n{n_nodes}_e{edge_prob}_p{p_layers}_r{repeat}.csv"

expt_cut = maxcut_bruteforce(graph)
rand_cut = random_partition_cut(graph)

@qml.qnode(dev)
def cost(params):
    gamma, beta = params[:p_layers], params[p_layers:]
    qaoa_circuit(gamma, beta, n_nodes, p_layers, H_C)
    return qml.expval(H_C)


with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Gamma", "Beta", "Energy", "Wall Time", "Ratio2Expt", "ImprovOvRand"])

    # Repeat experiment
    for i in range(repeat):
        np.random.seed(i)
        params = np.random.uniform(low=-np.pi, high=np.pi, size=2 * p_layers)
        for n in range(200):
            start_time = time.time()
            params, energy = opt.step_and_cost(cost, params)
            elapsed_time = time.time() - start_time
            
            # Write data to file
            writer.writerow([n, params[:p_layers], params[p_layers:], 
                            energy, elapsed_time, -energy / expt_cut,
                            -energy / rand_cut])