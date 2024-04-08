import numpy as np
import subprocess

# Parameters
vc_qubits_range = range(1, 11)
vc_layers_range = range(1, 11)
iterations = 500
repeat = 1

# Function to sample from a log-uniform distribution
def sample_log_uniform(min_val, max_val, samples):
    return np.exp(np.random.uniform(np.log(min_val), np.log(max_val), samples))

# Running the script with different parameters
for vc_qubits in vc_qubits_range:
    for vc_layers in vc_layers_range:
        vc_seed = vc_qubits * vc_layers
        command = f"python main.py --data RAND-VC --VC_qubits {vc_qubits} --VC_layers {vc_layers} " \
                    f"--VC_seed {vc_seed} --itr {iterations} --save_path results/randvc/ " \
                    f"--repeat {repeat} --VC_rand_ham --load_model RAND-VC-032e2829"
        # Execute the command
        subprocess.run(command, shell=True)