import subprocess
from tqdm import tqdm
molecules = {
    # 'H2': [0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.742, 0.78, 0.82, 0.86, 0.9, 0.94,
    #         0.98, 1.02, 1.06, 1.1, 1.14, 1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.46,
    #         1.5, 1.54, 1.58, 1.62, 1.66, 1.7, 1.74, 1.78, 1.82, 1.86, 1.9, 1.94, 1.98,
    #         2.02, 2.06, 2.1],
    # 'H3+': [0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82,
    #         0.86, 0.874, 0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.14,
    #         1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.46, 1.5,
    #         1.54, 1.58, 1.62, 1.66, 1.7, 1.74, 1.78, 1.82, 1.86,
    #         1.9, 1.94, 1.98, 2.02, 2.06, 2.1],
    # 'H4': [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66,
    #         0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84,
    #         0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02,
    #         1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2,
    #         1.22, 1.24, 1.26, 1.28, 1.3],
    # 'H5': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
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
'H5': [0.5, 0.6, 0.7, 0.8, 0.9,
       1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
}

for molname in ['H4']:
    for bond_length in tqdm(molecules[molname]):
        command = f"python main.py --data VQE-UCCSD --load_model \
                    RAND-VC-032e2829 --repeat 3 --molname {molname} \
                    --bond_length {bond_length} \
                    --save_path results/rale/ --verbose"
        if molname == 'H3+': command += ' --charge 1'
        subprocess.run(command, shell=True)