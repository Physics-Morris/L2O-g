import numpy as np
import matplotlib.pyplot as plt
from vqe_module import VQE_UCCSD_Problem
import pennylane as qml
from pennylane import numpy as pnp

def run_vqe(molname, bond_length, charge):
    # Initialize the problem with a specific bond length
    problem = VQE_UCCSD_Problem(molname=molname, bondlength=bond_length, charge=charge)

    # Get the loss function directly from the QNode
    loss_fn = problem.qnode

    # shape = qml.kUpCCGSD.shape(k=1, n_wires=problem.qubits, delta_sz=0)
    # params = pnp.zeros(shape)

    params = pnp.zeros(len(problem.singles) + len(problem.doubles))
    print('# of params', len(params))

    # Gradient descent optimizer
    optimizer = qml.GradientDescentOptimizer(stepsize=0.01)

    # Optimization loop, limited to a smaller number of steps for demonstration
    steps = 200
    all_loss = []
    for i in range(steps):
        params, energy = optimizer.step_and_cost(loss_fn, params)
        all_loss.append(energy)
        print("Epoch: {0:<5} | Test Loss: {1:<8.4f}".format(\
                i, energy), end="\r", flush=True)

    # Return the final loss, which corresponds to the energy
    return loss_fn(params), all_loss

def collect_fci_energies(molecule_name, bond_lengths, basis='STO-3G'):
    fci_energies = {}

    for bond_length in bond_lengths:
        # Load the dataset for the specific bond length
        dataset = qml.data.load('qchem', molname=molecule_name, bondlength=bond_length, basis=basis)[0]

        # Extract the FCI energy and store it in the dictionary
        fci_energy = dataset.fci_energy
        fci_energies[bond_length] = fci_energy

        # Optionally print the bond length and FCI energy
        print(f"Bond Length: {bond_length} Å, FCI Energy: {fci_energy} Ha")

    return fci_energies

def main():
    # H2
    # bond_lengths = [
    #     0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.742, 0.78, 0.82, 0.86, 0.9, 0.94,
    #     0.98, 1.02, 1.06, 1.1, 1.14, 1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.46,
    #     1.5, 1.54, 1.58, 1.62, 1.66, 1.7, 1.74, 1.78, 1.82, 1.86, 1.9, 1.94, 1.98,
    #     2.02, 2.06, 2.1
    # ]
    # H3+
    bond_lengths = [0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82,
                    0.86, 0.874, 0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.14,
                    1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.46, 1.5,
                    1.54, 1.58, 1.62, 1.66, 1.7, 1.74, 1.78, 1.82, 1.86,
                    1.9, 1.94, 1.98, 2.02, 2.06, 2.1]
    # H4
    bond_lengths = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66,
                    0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84,
                    0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02,
                    1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2,
                    1.22, 1.24, 1.26, 1.28, 1.3]

    # H5
    bond_lengths = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    # LiH
    # bond_lengths = [
    #     0.90, 0.93, 0.96, 0.99, 1.02, 1.05, 1.08, 1.11,
    #     1.14, 1.17, 1.20, 1.23, 1.26, 1.29, 1.32, 1.35,
    #     1.38, 1.41, 1.44, 1.47, 1.50, 1.53, 1.56, 1.57,
    #     1.59, 1.62, 1.65, 1.68, 1.71, 1.74, 1.77, 1.80,
    #     1.83, 1.86, 1.89, 1.92, 1.95, 1.98, 2.01, 2.04,
    #     2.07, 2.10,
    # ]
    # H2O
    # bond_lengths = [
    #       0.958
    # ]

    fci_energies = collect_fci_energies('H3+', bond_lengths)

    all_energies = []
    energies = []
    for bond_length in bond_lengths:
        energy, all_loss = run_vqe('H3+', bond_length, 1)
        all_energies.append(all_loss)
        energies.append(energy)
        print(f"Bond length {bond_length}: Final energy = {energy}")

    # Plotting the energies
    plt.figure(figsize=(10, 5))
    plt.plot(bond_lengths, energies, marker='o', linestyle='-', color='b')
    plt.title('Energy of H2 molecule as a function of bond length')
    plt.xlabel('Bond Length (Angstroms)')
    plt.ylabel('Energy (Hartree)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
