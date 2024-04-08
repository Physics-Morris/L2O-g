import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
from vqe_module import VQE_UCCSD_Problem
import json
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Function to select the optimizer based on name and learning rate
def select_optimizer(name, lr):
    if name == 'GD':
        return qml.GradientDescentOptimizer(stepsize=lr)
    elif name == 'Momentum':
        return qml.MomentumOptimizer(stepsize=lr)
    elif name == 'Adam':
        return qml.AdamOptimizer(stepsize=lr)
    elif name == 'Adagrad':
        return qml.AdagradOptimizer(stepsize=lr)
    elif name == 'RMSprop':
        return qml.RMSPropOptimizer(stepsize=lr)
    elif name == 'QNGD':
        return qml.QNGOptimizer(stepsize=lr, lam=0.01, approx="diag")
    else:
        raise ValueError(f"Unrecognized optimizer {name}")

def run_vqe(molname, bond_length, charge, optimizer_name, lr):
    problem = VQE_UCCSD_Problem(molname=molname, bondlength=bond_length, charge=charge)
    loss_fn = problem.qnode
    num_params = len(problem.singles) + len(problem.doubles)
    params = pnp.random.uniform(0, 2 * np.pi, num_params)
    fci = problem.get_fci()
    optimizer = select_optimizer(optimizer_name, lr)
    steps = 200
    all_loss = []
    for i in range(steps):
        params, energy = optimizer.step_and_cost(loss_fn, params)
        all_loss.append(float(energy))
    return float(loss_fn(params)), all_loss, fci

def process_job(args):
    molecule, bond_length, optimizer, lr = args
    charge = 1 if molecule == 'H3+' else 0
    energy, loss, fci = run_vqe(molecule, bond_length, charge, optimizer, lr)
    return molecule, bond_length, optimizer, lr, energy, loss, fci

def get_fci(molname, bond_length, charge):
    problem = VQE_UCCSD_Problem(molname=molname, bondlength=bond_length, charge=charge)
    fci = problem.get_fci()
    return fci


def main():
    molecules = {
        'H2': [0.5, 0.62, 0.7, 0.82, 0.9,
               1.02, 1.1, 1.22, 1.3, 1.42,
               1.5, 1.62, 1.7, 1.82, 1.9,
               2.02, 2.1],
        'H3+': [0.5, 0.62, 0.7, 0.82, 0.9,
                1.02, 1.1, 1.22, 1.3, 1.42,
                1.5, 1.62, 1.7, 1.82, 1.9,
                2.02, 2.1],
        'H4': [0.5, 0.6, 0.7, 0.8, 0.9,
               1.0, 1.1, 1.2, 1.26, 1.3],
        # 'H5': [0.5, 0.6, 0.7, 0.8, 0.9,
        #        1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        'LiH': [0.9, 0.93, 0.96, 0.99, 1.02,
                1.05, 1.08, 1.11, 1.14, 1.17,
                1.2, 1.23, 1.26, 1.29, 1.32,
                1.35, 1.38, 1.41, 1.44, 1.47,
                1.5, 1.53, 1.56, 1.57, 1.59,
                1.62, 1.65, 1.68, 1.71, 1.74,
                1.77, 1.8, 1.83, 1.86, 1.89,
                1.92, 1.95, 1.98, 2.01, 2.04,
                2.07, 2.1],
        'BeH2': [0.5 , 0.54, 0.58, 0.62,
                 0.66, 0.7 , 0.74, 0.78,
                 0.82, 0.86, 0.9 , 0.94,
                 0.98, 1.02, 1.06, 1.1 ,
                 1.14, 1.18, 1.22, 1.26,
                 1.3 , 1.33, 1.34, 1.38,
                 1.42, 1.46, 1.5 , 1.54,
                 1.58, 1.62, 1.66, 1.7 ,
                 1.74, 1.78, 1.82, 1.86,
                 1.9 , 1.94, 1.98, 2.02,
                 2.06, 2.1 ],
    }

    }
    # optimizers = ['GD', 'Momentum', 'Adam', 'Adagrad', 'RMSprop', 'QNGD']
    optimizers = ['GD', 'Adam']
    learning_rates = [0.01]
    jobs = [(molecule, bond_length, optimizer, lr) for molecule, bond_lengths in molecules.items()
            for bond_length in bond_lengths for optimizer in optimizers for lr in learning_rates]
    total_cnt = len(jobs)
    print('total jobs', total_cnt*5)

    cnt = 0
    restart = -1
    for run_num in range(11):
        # with ProcessPoolExecutor(max_workers=1) as executor:
            # for molecule, bond_length, optimizer, lr, energy, loss, fci in executor.map(process_job, jobs):
        for job in tqdm(jobs, desc='Runs'):
            cnt += 1
            if cnt >= restart:
                start_time = time.time()
                molecule, bond_length, optimizer, lr, energy, loss, fci = process_job(job)
                end_time = time.time()
                results_key = f"{molecule}_{optimizer}_{lr}_{bond_length}"
                with open(f"../results/uccsd_baselines_v3/{results_key}_energies_{run_num}.json", 'w') as f:
                    json.dump({'FCI': fci, 'Energy': energy, 'Loss': loss}, f)

                print(f"Completed: {molecule} at {bond_length}Å using {optimizer} with lr={lr}")
                print(f"{cnt}/{total_cnt*5}, time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
