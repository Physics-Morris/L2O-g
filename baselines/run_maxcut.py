import subprocess
import concurrent.futures

# Define parameters for combinations
optimizers = ['GD', 'Momentum', 'Adam', 'Adagrad', 'RMSprop', 'QNGD']
n_nodes_range = range(5, 11)  # From 5 to 10
edge_prob_range = [i / 10.0 for i in range(5, 11)]  # From 0.5 to 1.0
p_layers_range = range(1, 6)  # From 1 to 5
learning_rate = 0.01
seed = 0

# Command template
base_command = "python QAOA_MaxCut.py"

def build_command(optimizer, n_nodes, edge_prob, p_layers):
    """Construct the command string based on given parameters."""
    command = f"{base_command} --optimizer {optimizer} --n_nodes {n_nodes} --edge_prob {edge_prob} --lr {learning_rate} --p_layers {p_layers} --seed {seed} --repeat 1"
    return command

def run_command(command):
    """Function to execute a command via subprocess."""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    return process.stdout if process.stdout else process.stderr

# Prepare a list to hold all commands to execute
commands = [build_command(optimizer, n_nodes, edge_prob, p_layers)
            for optimizer in optimizers
            for n_nodes in n_nodes_range
            for edge_prob in edge_prob_range
            for p_layers in p_layers_range]

# Use ProcessPoolExecutor to run commands in parallel
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(run_command, commands))

# Optional: Print results
for result in results:
    print(result)