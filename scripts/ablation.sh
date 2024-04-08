# rand vc
python main.py --data RAND-VC --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --VC_qubits 7 --VC_layers 5 --itr 10
python main.py --data RAND-VC --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --VC_qubits 7 --VC_layers 8 --itr 10
python main.py --data RAND-VC --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --VC_qubits 10 --VC_layers 5 --itr 10

python main.py --data RAND-VC --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --VC_qubits 7 --VC_layers 5 --itr 10 --abl_g --abl_cl --preproc
python main.py --data RAND-VC --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --VC_qubits 7 --VC_layers 8 --itr 10 --abl_g --abl_cl --preproc
python main.py --data RAND-VC --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --VC_qubits 10 --VC_layers 5 --itr 10 --abl_g --abl_cl --preproc


# vqe hea
python main.py --data VQE-H2 --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --H2_bond 0.5
python main.py --data VQE-H2 --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --H2_bond 0.9
python main.py --data VQE-H2 --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --H2_bond 1.5

python main.py --data VQE-H2 --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --H2_bond 0.5 --abl_g --abl_cl --preproc
python main.py --data VQE-H2 --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --H2_bond 0.9 --abl_g --abl_cl --preproc
python main.py --data VQE-H2 --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --H2_bond 1.5 --abl_g --abl_cl --preproc

# vqe uccsd (dim=3,8,26)
python main.py --data VQE-UCCSD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname H2 --bond_length 0.5
python main.py --data VQE-UCCSD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname H3+ --bond_length 0.5 --charge 1
python main.py --data VQE-UCCSD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname H4 --bond_length 0.5
python main.py --data VQE-UCCSD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname H2 --bond_length 0.9
python main.py --data VQE-UCCSD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname H3+ --bond_length 0.9 --charge 1
python main.py --data VQE-UCCSD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname H4 --bond_length 0.9

python main.py --data VQE-UCCSD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname H2 --bond_length 0.5 --abl_g --abl_cl --preproc
python main.py --data VQE-UCCSD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname H2 --bond_length 0.9 --abl_g --abl_cl --preproc
python main.py --data VQE-UCCSD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname H3+ --bond_length 0.5 --charge 1 --abl_g --abl_cl --preproc
python main.py --data VQE-UCCSD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname H3+ --bond_length 0.9 --charge 1 --abl_g --abl_cl --preproc
python main.py --data VQE-UCCSD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname H4 --bond_length 0.5 --abl_g --abl_cl --preproc
# python main.py --data VQE-UCCSD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname H4 --bond_length 0.9 --abl_g --abl_cl --preproc


# vqe rale
python main.py --data VQE-RALE --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname LiH --bond_length 0.9
python main.py --data VQE-RALE --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname BeH2 --bond_length 0.9
python main.py --data VQE-RALE --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --molname H2O --bond_length 0.9

python main.py --data VQE-RALE --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname LiH --bond_length 0.9 --abl_g --abl_cl --preproc
python main.py --data VQE-RALE --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname BeH2 --bond_length 0.9 --abl_g --abl_cl --preproc
python main.py --data VQE-RALE --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --molname H2O --bond_length 0.9 --abl_g --abl_cl --preproc


# maxcut
python main.py --data QAOA_MaxCut --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 3 --QAOA_edge_prob 0.5
python main.py --data QAOA_MaxCut --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 3 --QAOA_edge_prob 0.6
python main.py --data QAOA_MaxCut --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 3 --QAOA_edge_prob 0.7

python main.py --data QAOA_MaxCut --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 3 --QAOA_edge_prob 0.5 --abl_g --abl_cl --preproc
python main.py --data QAOA_MaxCut --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 3 --QAOA_edge_prob 0.6 --abl_g --abl_cl --preproc
python main.py --data QAOA_MaxCut --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 3 --QAOA_edge_prob 0.7 --abl_g --abl_cl --preproc


# sk
python main.py --data QAOA_MaxCut --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 1 --QAOA_edge_prob 1.0
python main.py --data QAOA_MaxCut --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 3 --QAOA_edge_prob 1.0
python main.py --data QAOA_MaxCut --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 5 --QAOA_edge_prob 1.0

python main.py --data QAOA_MaxCut --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 1 --QAOA_edge_prob 1.0 --abl_g --abl_cl --preproc
python main.py --data QAOA_MaxCut --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 3 --QAOA_edge_prob 1.0 --abl_g --abl_cl --preproc
python main.py --data QAOA_MaxCut --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --QAOA_n_nodes 6 --QAOA_p_layers 5 --QAOA_edge_prob 1.0 --abl_g --abl_cl --preproc


# reupload
python main.py --data REUPLOAD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --RP_layers 3
python main.py --data REUPLOAD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --RP_layers 5
python main.py --data REUPLOAD --load_model RAND-VC-032e2829 --save_path results/ablation/ --verbose --repeat 5 --RP_layers 8

python main.py --data REUPLOAD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --RP_layers 3 --abl_g --abl_cl --preproc
python main.py --data REUPLOAD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --RP_layers 5 --abl_g --abl_cl --preproc
python main.py --data REUPLOAD --load_model ablation/RAND-VC-lstm --save_path results/ablation/ --verbose --repeat 5 --RP_layers 8 --abl_g --abl_cl --preproc
