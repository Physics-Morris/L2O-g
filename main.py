import argparse
import os
import torch
import wandb
import numpy
import yaml
from tqdm import tqdm

from utils.wandb_config import _login, _sweep_config, _log_config
from exp.exp_l2o_g import exp_l2o_g_NG
from utils.tools import generate_model_name, load_problem

def main():
    parser = argparse.ArgumentParser(description='l2o-g')

    # train arg
    parser.add_argument('--train', default=False, action='store_true', help='train optimizer')
    parser.add_argument('--verbose', default=False, action='store_true', help='verbose')
    parser.add_argument('--data', type=str, default='VQE-H2', help='task')
    parser.add_argument('--load_model', type=str, default='None', help='model name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model')
    parser.add_argument('--seed', type=int, default=0, help='Seed')

    # hpo arg
    parser.add_argument('--preproc', action='store_false', help='preprocess', default=True)
    parser.add_argument('--preproc_factor', type=int, default=10, help='preprocess factor')
    parser.add_argument('--hidden_sz', type=int, default=30, help='hidden size')
    parser.add_argument('--lamb_a', type=float, default=0.01, help='lambda a')
    parser.add_argument('--lamb_b', type=float, default=0.01, help='lambda b')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--ep', type=int, default=20, help='trained optimzer epochs')
    parser.add_argument('--period', type=int, default=5, help='Number of period for each epochs')

    parser.add_argument('--itr', type=int, default=200, help='optimzer iteration')
    parser.add_argument('--repeat', type=int, default=10, help='repeat exp n times')
    parser.add_argument('--n_tests', type=int, default=100, help='number of repeat in testing')


    # gpu arg
    parser.add_argument('--use_gpu', action='store_true', help='use gpus', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # wandb arg
    parser.add_argument('--record', default=False, action='store_true', help='Wandb Record')
    parser.add_argument('--project', type=str, default='l2o-g', help='project name')
    parser.add_argument('--sweep', default=False, action='store_true', help='HPO Sweep')

    # VQE problem argument
    parser.add_argument("--H2_bond", type=float, default=0.7, help="Bond length for the VQE-H2 problem")
    parser.add_argument("--molname", type=str, default='H2', help="Name of the mol")
    parser.add_argument("--bond_length", type=float, default=0.7, help="Bond length for the VQE-UCSSD problem")
    parser.add_argument('--charge', type=int, default=0, help='charge of mol')
    parser.add_argument('--rale_layers', type=int, default=1, help='# of layers in RALE')

    # Rand_VC problem argument
    parser.add_argument('--VC_qubits', type=int, default=4, help='Number of qubits in random VC problem')
    parser.add_argument('--VC_layers', type=int, default=1, help='Number of layers in random VC problem')
    parser.add_argument('--VC_seed', type=int, default=0, help='Seed use for rand VC problem')
    parser.add_argument('--VC_rand_ham', default=False, action='store_true', help='random hamiltonian')

    # Reupload Circuit
    parser.add_argument('--RP_func_type', type=str, default='circle', help='classifer function type')
    parser.add_argument('--RP_qubits', type=int, default=1, help='Number of qubits in random VC problem')
    parser.add_argument('--RP_layers', type=int, default=3, help='Number of layers in random VC problem')
    parser.add_argument('--RP_ndata', type=int, default=5000, help='Number of total data')
    parser.add_argument('--RP_seed', type=int, default=0, help='Seed use for rand VC problem')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    # Dressed Circuit
    parser.add_argument('--DR_tasks', type=str, default='plane_car', help='task type in cifar10')
    parser.add_argument('--DR_qubits', type=int, default=4, help='Number of qubits in random DR problem')
    parser.add_argument('--DR_quantum', default=True, action='store_false', help='Use quantum')
    parser.add_argument('--DR_c_model', type=str, default='512_n', help='type of pretrain classical model')
    parser.add_argument('--DR_q_depth', type=int, default=4, help='quantum depth')
    parser.add_argument('--DR_max_layers', type=int, default=15, help='max layers')
    parser.add_argument('--DR_q_delta', type=float, default=0.1, help='initial parameter distribution')
    parser.add_argument('--DR_seed', type=int, default=0, help='DR seed')

    # QAOA MaxCut
    parser.add_argument('--QAOA_n_nodes', type=int, default=3, help='Number of node in QAOA MaxCut')
    parser.add_argument('--QAOA_p_layers', type=int, default=1, help='Number of layers in QAOA MaxCut')
    parser.add_argument('--QAOA_edge_prob', type=float, default=0.5, help='QAOA edge probability')
    parser.add_argument('--QAOA_seed', type=int, default=0, help='QAOA seed')

    # test option
    parser.add_argument('--save_path', type=str, default='results/', help='save path')
    parser.add_argument('--save_traj', default=False, action='store_true', help='save parameter trajectory')

    # ablation option
    parser.add_argument('--abl_cl', default=False, action='store_true', help='Ablation on CL')
    parser.add_argument('--abl_g', default=False, action='store_true', help='Ablation on g')


    args = parser.parse_args()

    if args.abl_cl:
        cl = [20]*8
    else:
        cl = [10, 20, 40, 60, 80, 100, 120, 150]

    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    if args.record: _login()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.sweep: _sweep_config(args)

    print('Args in experiment:')
    print(args)

    Exp = exp_l2o_g_NG

    setting = generate_model_name(args)

    if not args.sweep and args.record: _log_config(args, setting)

    dataset = None
    qnode = None
    dressed_model = None
    if args.data == 'REUPLOAD':
        dimension, cost_func, metric_fn, qnode, dataset = load_problem(args)
    elif args.data == 'DRESSED':
        dimension, cost_func, metric_fn, qnode, dressed_model = load_problem(args)
    else:
        dimension, cost_func, metric_fn = load_problem(args)

    if args.train:
        exp = Exp(args, dimension, cost_func, metric_fn, cl,
                  args.data=='REUPLOAD', dataset, args.batch_size, qnode,
                  args.data=='DRESSED', args.DR_q_delta, dressed_model,
                  args.abl_g)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        best_loss, best_model = exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.model.load_state_dict(best_model)
        loss, _ = exp.test(exp.model, args.itr)
        print("Test Loss: {0:.3f}".format(loss[-1]))
    else:
        exp = Exp(args, dimension, cost_func, metric_fn, cl, args.data=='REUPLOAD',
                  dataset, args.batch_size, qnode, args.data=='DRESSED',
                  args.DR_q_delta, dressed_model, args.abl_g)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        path = os.path.join(args.checkpoints, args.load_model, 'model.pth')
        state_dict = torch.load(path)
        exp.model.load_state_dict(state_dict)
        all_loss = {}
        for i in tqdm(range(args.repeat)):
            if args.data == 'REUPLOAD':
                loss, acc, f1 = exp.test(exp.model, args.itr, exp.x_train,
                                         exp.y_train, exp.x_test, exp.y_test,
                                         args.verbose)
                all_loss[f'loss_{i}'] = loss
                all_loss[f'acc_{i}'] = acc
                all_loss[f'f1_{i}'] = f1
            elif args.data == 'DRESSED':
                loss, acc, acc_train = exp.test(exp.model, args.itr, verbose=args.verbose)
                all_loss[f'loss_{i}'] = loss
                all_loss[f'acc_{i}'] = acc
                all_loss[f'acc_train{i}'] = acc_train
            else:
                if args.save_traj:
                    loss, traj, wall_time = exp.test(exp.model, args.itr, save_traj=args.save_traj)
                    all_loss[f'traj_{i}'] = traj
                else:
                    loss, wall_time = exp.test(exp.model, args.itr,
                                               save_traj=args.save_traj,
                                               verbose=args.verbose)
                all_loss[f'test_{i}'] = loss
                all_loss[f'time_{i}'] = wall_time

        save_path = args.save_path + 'mod_' + args.load_model + '_set_' + setting + '/'
        if not os.path.exists(save_path): os.makedirs(save_path)

        # Convert the args to a dictionary
        args_dict = vars(args)
        arg_save_path = save_path + 'args.yaml'
        # Save the dictionary to a YAML file
        with open(arg_save_path, 'w') as file:
            yaml.dump(args_dict, file, default_flow_style=False)

        numpy.savez_compressed(save_path + 'test.npz', **all_loss)
        print('Saving result to...', save_path)

    torch.cuda.empty_cache()
    wandb.finish()

main()
