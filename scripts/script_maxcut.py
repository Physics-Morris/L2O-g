import subprocess
import concurrent.futures

def run_command(command):
    """Function to execute a command via subprocess."""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    result = process.stdout
    if process.stderr:
        result += "\nError:" + process.stderr
    return result

# Parameters ranges
n_nodes_range = range(5, 11)  # From 5 to 10
p_layers_range = [1, 5]
edge_prob_range = [i / 10.0 for i in range(5, 11)]  # From 0.5 to 1.0

# Command template
base_command = "python main.py --data QAOA_MaxCut"

# List to hold all commands to execute
commands = []

# Build all commands
for n_nodes in n_nodes_range:
    for p_layers in p_layers_range:
        for edge_prob in edge_prob_range:
            command = f"{base_command} --QAOA_n_nodes {n_nodes} --QAOA_p_layers {p_layers} --QAOA_edge_prob {edge_prob} --itr 200 --load_model QAOA_MaxCut-0b785a1f --repeat 5"
            commands.append(command)

# Use ProcessPoolExecutor to run commands in parallel
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Map the run_command function to the commands, executing them concurrently
    results = list(executor.map(run_command, commands))

# Print results
for result in results:
    print(result)