import os
import wandb

def _login():
    key_file_path = 'utils/wandb_api_key.txt' 
    with open(key_file_path, 'r') as key_file:
        key = key_file.read().strip()
    os.environ["WANDB_API_KEY"] = key
    wandb.login()

def _sweep_config(args):
    wandb.init(project=args.project, config={})
    
    args.seed = wandb.config.get('seed', args.seed)
    args.preproc = wandb.config.get('preproc', args.preproc)
    args.preproc_factor = wandb.config.get('preprocess_factor', args.preproc_factor)
    args.hidden_sz = wandb.config.get('hidden_sz', args.hidden_sz)
    args.lamb_a = wandb.config.get('lamb_a', args.lamb_a)
    args.lamb_b = wandb.config.get('lamb_b', args.lamb_b)
    args.lr = wandb.config.get('lr', args.lr)
    args.ep = wandb.config.get('ep', args.ep)
    args.period = wandb.config.get('period', args.period)
    args.n_tests = wandb.config.get('n_tests', args.n_tests)
    args.itr = wandb.config.get('itr', args.itr)
    
    args.use_multi_gpu = wandb.config.get('use_multi_gpu', args.use_multi_gpu)
    args.devices = wandb.config.get('devices', args.devices)
    
    args.record = wandb.config.get('record', args.record)
    args.sweep = wandb.config.get('sweep', args.sweep)
    
    args.dataset = wandb.config.get('data', args.data)
    args.H2_bond = wandb.config.get('H2_bond', args.H2_bond)


def _log_config(args, setting):
    wandb.init(
        project=args.project,
        config={
            "model": "l2o-g",  
            "seed": args.seed,
            "preproc": args.preproc,
            "preprocess_factor": args.preproc_factor,
            "hidden_sz": args.hidden_sz,
            "lamb_a": args.lamb_a,
            "lamb_b": args.lamb_b,
            "lr": args.lr,
            "ep": args.ep,
            "period": args.period,
            "n_tests": args.n_tests,
            "itr": args.itr,
            "use_multi_gpu": args.use_multi_gpu,
            "devices": args.devices,
            "record": args.record,
            "sweep": args.sweep,
            "dataset": args.data, 
            "H2_bond": args.H2_bond,
            "VC_qubits": args.VC_qubits,
            "VC_layers": args.VC_layers,
            "VC_seed": args.VC_seed,
            "saved_model": setting,
            "train": args.train
        }
    )
