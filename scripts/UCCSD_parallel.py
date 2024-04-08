import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Define your molecules and their bond lengths
molecules = {
    'H2': [0.5, 0.62, 0.7, 0.82, 0.9, 1.02, 1.1, 1.22, 1.3, 1.42,
           1.5, 1.62, 1.7, 1.82, 1.9, 2.02, 2.1],
    'H3+': [0.5, 0.62, 0.7, 0.82, 0.9, 1.02, 1.1, 1.22, 1.3, 1.42,
            1.5, 1.62, 1.7, 1.82, 1.9, 2.02, 2.1],
    'H4': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.26, 1.3]
}

# Function to run a subprocess command
def run_simulation(molname, bond_length):
    command = f"python main.py --data VQE-UCCSD --load_model RAND-VC-032e2829 --repeat 100 " \
              f"--molname {molname} --bond_length {bond_length} --save_path results/uccsd_l2o_g/"
    if molname == 'H3+':
        command += ' --charge 1'
    subprocess.run(command, shell=True)

# Use ProcessPoolExecutor to distribute the jobs
def distribute_jobs(molecules):
    with ProcessPoolExecutor() as executor:
        futures = []
        for molname, bond_lengths in molecules.items():
            for bond_length in bond_lengths:
                futures.append(executor.submit(run_simulation, molname, bond_length))

        # Optional: Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running simulations"):
            future.result()  # Wait for completion and handle exceptions if necessary

# Call the function to distribute jobs
if __name__ == "__main__":
    distribute_jobs(molecules)
