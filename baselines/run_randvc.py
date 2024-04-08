import numpy as np
import subprocess
import multiprocessing

# Parameters
vc_qubits_range = range(1, 11)
vc_layers_range = range(1, 11)
optimizers = ['gd', 'qngd', 'adam', 'mom']
iterations = 200
repeat = 1
lr_min = 1e-5
lr_max = 1e-1
lr_samples = 20

lrs = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

# Function to sample from a log-uniform distribution
def sample_log_uniform(min_val, max_val, samples):
    return np.exp(np.random.uniform(np.log(min_val), np.log(max_val), samples))

# Function to execute command in subprocess
def execute_command(command):
    subprocess.run(command, shell=True)

# Define the main function for distributing the runs
def distribute_runs():
    # Calculate the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Calculate the number of processes to use for 80% of the CPU
    num_processes = int(num_cores * 0.8)

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Running the script with different parameters
    for vc_qubits in vc_qubits_range:
        for vc_layers in vc_layers_range:
            vc_seed = vc_qubits * vc_layers
            for opt in optimizers:
                # lrs = sample_log_uniform(lr_min, lr_max, lr_samples)
                for lr in lrs:
                    if opt == 'gd': iterations = 200 * 30
                    else: iterations = 200
                    command = f"python RANDVC.py --VC_qubits {vc_qubits} --VC_layers {vc_layers} " \
                              f"--VC_seed {vc_seed} --opt {opt} --lr {lr} --itr {iterations} " \
                              f"--repeat {repeat}"
                    # Execute the command in a separate process
                    pool.apply_async(execute_command, args=(command,))
    
    # Close the pool to release resources
    pool.close()
    pool.join()

if __name__ == "__main__":
    distribute_runs()