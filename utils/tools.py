import sys
import functools
from torch.autograd import Variable
import hashlib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils.meta_module import *
from problems.VQE import VQE_H2_Problem
from problems.VQE_UCCSD import VQE_UCCSD_Problem
from problems.VQE_RALE import VQE_RALE_Problem
from problems.RANDVC import RANDVC_problem
from problems.REUPLOAD import REUPLOAD_problem
from problems.DRESSED import DRESSED_problem
from problems.QAOA import QAOA_problem


def detach_var(v):
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def generate_model_name(args):
    base_string = f"{args.data}_{args.preproc}_{args.hidden_sz}_{args.lamb_a}_\
                    {args.lamb_b}_{args.lr}_{args.ep}_{args.period}_{args.n_tests}_\
                    {args.itr}_{args.use_multi_gpu}_{args.seed}_{args.H2_bond}_\
                    {args.bond_length}_{args.molname}_{args.charge}_{args.rale_layers}_\
                    {args.VC_qubits}_{args.VC_layers}_{args.VC_seed}_{args.VC_rand_ham}_\
                    {args.RP_func_type}_{args.RP_qubits}_{args.RP_layers}_{args.RP_ndata}_\
                    {args.RP_seed}_{args.batch_size}_{args.DR_tasks}_{args.DR_qubits}_\
                    {args.DR_quantum}_{args.DR_c_model}_{args.DR_q_depth}_{args.DR_max_layers}_\
                    {args.DR_max_layers}_{args.DR_q_delta}_{args.DR_seed}_{args.QAOA_n_nodes}_\
                    {args.QAOA_p_layers}_{args.QAOA_edge_prob}_{args.QAOA_seed}_{args.abl_cl}_\
                    {args.abl_g}"

    unique_hash = hashlib.sha1(base_string.encode()).hexdigest()[:8]

    model_name = f"{args.data}-{unique_hash}"

    return model_name

def load_problem(args):
    if args.data == 'VQE-H2':
        print('Running Problem VQE-H2')
        problem = VQE_H2_Problem(bondlength=args.H2_bond)
        dimension = 12
        cost_func = problem.get_loss_function()
        metric_fn = problem.get_metric_fn()
        return dimension, cost_func, metric_fn
    elif args.data == 'VQE-UCCSD':
        print('Running Problem VQE-UCCSD')
        problem = VQE_UCCSD_Problem(molname=args.molname, bondlength=args.bond_length,
                                    charge=args.charge)
        dimension = problem.get_dimension()
        print('Dimension of the problem', dimension)
        cost_func = problem.get_loss_function()
        metric_fn = problem.get_metric_fn()
        return dimension, cost_func, metric_fn
    elif args.data == 'VQE-RALE':
        print('Running Problem VQE-RALE')
        problem = VQE_RALE_Problem(molname=args.molname, bondlength=args.bond_length,
                                    charge=args.charge, layers=args.rale_layers)
        dimension = problem.get_dimension()
        print('Dimension of the problem', dimension)
        cost_func = problem.get_loss_function()
        metric_fn = problem.get_metric_fn()
        return dimension, cost_func, metric_fn
    elif args.data == 'RAND-VC':
        print('Running Problem RAND-VC')
        problem = RANDVC_problem(qubits=args.VC_qubits, layers=args.VC_layers,
                                 seed=args.VC_seed, rand_ham=args.VC_rand_ham)
        dimension = args.VC_qubits * args.VC_layers
        cost_func = problem.get_loss_function()
        metric_fn = problem.get_metric_fn()
        return dimension, cost_func, metric_fn
    elif args.data == 'REUPLOAD':
        print('Running Problem REUPLOAD')
        problem = REUPLOAD_problem(func_type=args.RP_func_type, qubits=args.RP_qubits, layers=args.RP_layers,
                                   splits=[.7, .2, .1], n_data=args.RP_ndata, batch_size=args.batch_size,
                                   seed=args.RP_seed)
        dimension = args.RP_qubits * args.RP_layers * 3
        x_train, y_train, x_val, y_val, x_test, y_test = problem.get_data()
        cost_func = problem.get_loss_function()
        metric_fn = problem.get_metric_fn()
        qnode = problem.get_qnode()
        return dimension, cost_func, metric_fn, qnode, [x_train, y_train, x_val, y_val, x_test, y_test]
    elif args.data == 'DRESSED':
        print('Running Problem DRESSED')
        if args.DR_tasks == 'plane_car': task = ['plane', 'car']
        elif args.DR_tasks == 'dog_cat': task = ['dog', 'cat']
        else: print('Unrecog. DRESSED task', args.DR_task)
        problem = DRESSED_problem(task=task, n_qubits=args.DR_qubits, quantum=args.DR_quantum, c_model=args.DR_c_model,
                                  batch_size=args.batch_size, q_depth=args.DR_q_depth, max_layers=args.DR_max_layers,
                                  q_delta=args.DR_q_delta, rng_seed=args.DR_seed, device='cpu')
        dimension = args.DR_max_layers * args.DR_qubits
        cost_func = problem.get_loss_function()
        metric_fn = problem.get_metric_fn()
        qnode = problem.get_qnode()
        dressed_model = problem.get_dressed_model()
        return dimension, cost_func, metric_fn, qnode, dressed_model
    elif args.data == 'QAOA_MaxCut':
        print('Running Problem QAOA_MaxCut')
        problem = QAOA_problem(n_nodes=args.QAOA_n_nodes, p_layers=args.QAOA_p_layers, edge_prob=args.QAOA_edge_prob,
                               seed=args.QAOA_seed, qaoa_type='MaxCut')
        dimension = args.QAOA_p_layers * 2
        cost_func = problem.get_loss_function()
        metric_fn = problem.get_metric_fn()
        return dimension, cost_func, metric_fn
    else:
        print('Data not implemented')
        dimension, cost_func, metric_fn = None, None, None
        sys.exit()

def classification_metrics(y_true, y_pred, y_score=None):
    """
    Calculate various classification metrics.

    Args:
        y_true (array-like): True labels of the data.
        y_pred (array-like): Predicted labels of the data.
        y_score (array-like, optional): Target scores, can either be probability estimates of the positive class,
                                        confidence values, or non-thresholded measure of decisions
                                        (as returned by “decision_function” on some classifiers).

    Returns:
        dict: A dictionary containing 'accuracy', 'auc' (if y_score is provided), and 'f1' scores.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary')  # Use 'micro', 'macro', 'samples','weighted' for multiclass
    }

    if y_score is not None:
        metrics['auc'] = roc_auc_score(y_true, y_score)

    return metrics
