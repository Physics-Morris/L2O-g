import numpy as np
from vqe_module import VQE_UCCSD_Problem
from tqdm import tqdm

def get_energies(molname, bond_length, charge):
    """Retrieve both HF and FCI energies for a given molecule and bond length."""
    problem = VQE_UCCSD_Problem(molname=molname, bondlength=bond_length, charge=charge)
    fci_energy = problem.get_fci()
    return fci_energy

def gather_energies():
    molecules = {
        # 'H2': [0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.742, 0.78, 0.82, 0.86, 0.9, 0.94,
        #        0.98, 1.02, 1.06, 1.1, 1.14, 1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.46,
        #        1.5, 1.54, 1.58, 1.62, 1.66, 1.7, 1.74, 1.78, 1.82, 1.86, 1.9, 1.94, 1.98,
        #        2.02, 2.06, 2.1],
        # 'H3+': [0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82,
        #         0.86, 0.874, 0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.14,
        #         1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.46, 1.5,
        #         1.54, 1.58, 1.62, 1.66, 1.7, 1.74, 1.78, 1.82, 1.86,
        #         1.9, 1.94, 1.98, 2.02, 2.06, 2.1],
        # 'H4': [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66,
        #        0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84,
        #        0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02,
        #        1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2,
        #        1.22, 1.24, 1.26, 1.28, 1.3],
        # 'LiH': [0.9, 0.93, 0.96, 0.99, 1.02,
        #         1.05, 1.08, 1.11, 1.14, 1.17,
        #         1.2, 1.23, 1.26, 1.29, 1.32,
        #         1.35, 1.38, 1.41, 1.44, 1.47,
        #         1.5, 1.53, 1.56, 1.57, 1.59,
        #         1.62, 1.65, 1.68, 1.71, 1.74,
        #         1.77, 1.8, 1.83, 1.86, 1.89,
        #         1.92, 1.95, 1.98, 2.01, 2.04,
        #         2.07, 2.1],
        # 'BeH2': [0.5 , 0.54, 0.58, 0.62,
        #          0.66, 0.7 , 0.74, 0.78,
        #          0.82, 0.86, 0.9 , 0.94,
        #          0.98, 1.02, 1.06, 1.1 ,
        #          1.14, 1.18, 1.22, 1.26,
        #          1.3 , 1.33, 1.34, 1.38,
        #          1.42, 1.46, 1.5 , 1.54,
        #          1.58, 1.62, 1.66, 1.7 ,
        #          1.74, 1.78, 1.82, 1.86,
        #          1.9 , 1.94, 1.98, 2.02,
        #          2.06, 2.1 ],
        'H2O': [0.5   , 0.54, 0.58,
                0.62  , 0.66, 0.7 ,
                0.74  , 0.78, 0.82,
                0.86  , 0.9 , 0.94,
                0.958 , 0.98, 1.02,
                1.06  , 1.1 , 1.14,
                1.18  , 1.22, 1.26,
                1.3   , 1.34, 1.38,
                1.42  , 1.46, 1.5 ,
                1.54  , 1.58, 1.62,
                1.66  , 1.7 , 1.74,
                1.78  , 1.82, 1.86,
                1.9   , 1.94, 1.98,
                2.02  , 2.06, 2.1 ]
    }
    fci_results = {}

    for molname, bond_lengths in molecules.items():
        fci_energies = []
        print(molname)
        for bond_length in tqdm(bond_lengths):
            charge = 1 if molname == 'H3+' else 0
            fci_energy = get_energies(molname, bond_length, charge)
            fci_energies.append(fci_energy)
        fci_results[molname] = fci_energies

    # Saving results to npy files
    for molname in molecules.keys():
        np.save(f'results/fci_energies_{molname}.npy', np.array(fci_results[molname]))

    return fci_results

if __name__ == "__main__":
    fci_results = gather_energies()
    print("HF and FCI energies have been calculated and saved.")
