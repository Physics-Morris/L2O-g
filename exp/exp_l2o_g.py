import os

from exp.exp_basic import Exp_Basic
from models.l2o_g import L2O_g
from utils.tools import rgetattr, detach_var, rsetattr, classification_metrics

import tqdm
import copy
import torch
import torch.nn as nn
from torch import optim
from torch.nn import DataParallel
import torch.utils.data as data_utils
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import numpy as np

import os
import time
import json
import pickle
import wandb

import warnings
warnings.filterwarnings('ignore')

from utils.meta_module import *


def get_optimizee(dimension, op_type=None, x=None, y=None, batch_size=None, q_delta=None, model=None):
    class Optimizee(MetaModule):
        def __init__(self, dimension=12):
            super().__init__()
            self.register_buffer('theta', to_var(torch.rand(dimension) * 2 * np.pi,
                                                 requires_grad=True))
            # self.register_buffer('theta', to_var(torch.zeros(dimension), requires_grad=True))
            # custome_init = torch.tensor([0.15, 0.4])
            # self.register_buffer('theta', to_var(custome_init, requires_grad=True))

        def forward(self, loss_function):
            return loss_function.get_loss(self.theta.cpu())

        def all_named_parameters(self):
            return [('theta', self.theta)]

    class REUPLOAD_Optimizee(MetaModule):
        def __init__(self, x, y, dimension):
            super().__init__()
            self.register_buffer('theta', to_var(torch.rand(dimension) * 2 * np.pi,
                                                 requires_grad=True))
            self.x = x
            self.y = y

        def forward(self, loss_function):
            params = [self.theta.cpu(), self.x, self.y]
            return loss_function.get_loss(params)

        def all_named_parameters(self):
            return [('theta', self.theta)]

    class DRESSED_Optimizee(MetaModule):
        def __init__(self, dimension, q_delta, model):
            super().__init__()
            self.register_buffer('theta', to_var(torch.randn(dimension) * q_delta,
                                                 requires_grad=True))
            self.model = model

        def forward(self, loss_function, phase):
            self.model.fc.q_params.data = self.theta.data
            loss = loss_function.get_loss(self.model, phase)
            return loss

        def all_named_parameters(self):
            return [('theta', self.theta), *self.model.named_parameters()]

    if op_type == 'REUPLOAD':
        return REUPLOAD_Optimizee(x=x, y=y, dimension=dimension)
    elif op_type == 'DRESSED':
        return DRESSED_Optimizee(dimension=dimension, q_delta=q_delta, model=model)
    else:
        return Optimizee(dimension=dimension)

class exp_l2o_g_NG(Exp_Basic):
    def __init__(self, args, dimension, cost_func, metric_fn, cl, REUPLOAD=False,
                dataset=None, batch_size=None, qnode=None, DRESSED=None,
                q_delta=None, dressed_model=None, abl_g=False):
        super(exp_l2o_g_NG, self).__init__(args)
        self.dimension = dimension
        self.cost_func = cost_func
        self.metric_fn = metric_fn
        self.cl = cl
        self.abl_g = abl_g
        self.model = self._build_model().to(self.device)
        self.REUPLOAD = REUPLOAD
        self.DRESSED = DRESSED
        self.dataset = dataset
        self.batch_size = batch_size
        self.qnode = qnode
        self.x_train = None
        self.y_train = None
        self.x_val   = None
        self.y_val   = None
        self.x_test  = None
        self.y_test  = None
        if self.dataset is not None:
            [x_train, y_train, x_val, y_val, x_test, y_test] = self.dataset
            self.x_train = x_train
            self.y_train = y_train
            self.x_val   = x_val
            self.y_val   = y_val
            self.x_test  = x_test
            self.y_test  = y_test
        # Dressed circuit
        self.q_delta = q_delta
        self.dressed_model = dressed_model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = L2O_g(metric_fn=self.metric_fn,
                        preproc=self.args.preproc,
                        hidden_sz=self.args.hidden_sz,
                        preproc_factor=self.args.preproc_factor,
                        lamb_a=self.args.lamb_a,
                        lamb_b=self.args.lamb_b,
                        device=self.device,
                        abl_g=self.abl_g)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of parameters in the L2O-g+: {params}")
        return model

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.lr)

    def _model_step(self, unroll, optim_it, x=None, y=None):
        self.model.train()
        model_optim = self._select_optimizer()

        ### Optimizee dependent ###
        if self.REUPLOAD: optimizee = get_optimizee(self.dimension, op_type='REUPLOAD', x=x, y=y)
        else: optimizee = get_optimizee(self.dimension)
        ### Optimizee dependent ###

        target = self.cost_func
        n_params = 0
        for name, p in optimizee.all_named_parameters():
            n_params += int(np.prod(p.size()))
        hidden_states = [Variable(torch.zeros(n_params, self.model.hidden_sz)) for _ in range(2)]
        cell_states = [Variable(torch.zeros(n_params, self.model.hidden_sz)) for _ in range(2)]
        all_losses_ever = []
        model_optim.zero_grad()
        all_losses = None
        for iteration in range(1, optim_it + 1):
            loss = optimizee(target)

            ### Optimizee dependent ###
            if self.REUPLOAD:
                y_predict, fidel = evaluate_REUPLOAD(optimizee.all_named_parameters()[0][1].detach().cpu(), x, target, self.qnode)
                metrics = classification_metrics(y, y_predict)
                if self.args.record: wandb.log({"Train Inner ACC": metrics['accuracy'], "Train Inner F1": metrics['f1']})
            ### Optimizee dependent ###

            if self.args.record: wandb.log({"Train Inner Loss": loss.data.cpu().numpy()})

            if all_losses is None:
                all_losses = loss
            else:
                all_losses += loss

            all_losses_ever.append(loss.data.cpu().numpy())
            loss.backward(retain_graph=True)

            offset = 0
            result_params = {}
            hidden_states2 = [Variable(torch.zeros(n_params, self.model.hidden_sz)) for _ in range(2)]
            cell_states2 = [Variable(torch.zeros(n_params, self.model.hidden_sz)) for _ in range(2)]

            for name, p in optimizee.all_named_parameters():
                cur_sz = int(np.prod(p.size()))
                gradients = detach_var(p.grad.view(cur_sz, 1))
                updates, new_hidden, new_cell = self.model(
                    p, gradients,
                    [h[offset:offset+cur_sz] for h in hidden_states],
                    [c[offset:offset+cur_sz] for c in cell_states]
                )
                for i in range(len(new_hidden)):
                    hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                    cell_states2[i][offset:offset+cur_sz] = new_cell[i]
                result_params[name] = p + updates
                result_params[name].retain_grad()

                offset += cur_sz

            if iteration % unroll == 0 or iteration == optim_it:
                model_optim.zero_grad()
                all_losses.backward()
                model_optim.step()

                all_losses = None

                ### Optimizee dependent ###
                if self.REUPLOAD: optimizee = get_optimizee(self.dimension, op_type='REUPLOAD', x=x, y=y)
                else: optimizee = get_optimizee(self.dimension)
                optimizee.load_state_dict(result_params)
                optimizee.zero_grad()
                ### Optimizee dependent ###

                hidden_states = [detach_var(v) for v in hidden_states2]
                cell_states = [detach_var(v) for v in cell_states2]

            else:
                for name, p in optimizee.all_named_parameters():
                    rsetattr(optimizee, name, result_params[name])
                assert len(list(optimizee.all_named_parameters()))
                hidden_states = hidden_states2
                cell_states = cell_states2
        return all_losses_ever


    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'model.pth'
        if not os.path.exists(path): os.makedirs(path)
        best_model = None
        best_loss  = np.inf

        i = 0
        while True:
            n = 0
            stop = True
            while (n < self.args.ep) or (val_loss == best_loss):
                n = n + 1
                all_loss = self._model_step(unroll=self.cl[i], optim_it=self.cl[i]*self.args.period, x=self.x_train, y=self.y_train)

                val_loss, test_acc, test_f1 = [], [], []
                for _ in range(self.args.n_tests):
                    if self.REUPLOAD:
                        one_val_loss, acc, f1 = self.test(model=self.model, optim_it=self.cl[i+1], x_train=self.x_train,
                                                          y_train=self.y_train, x_test=self.x_test, y_test=self.y_test)
                        test_acc.append(acc[-1])
                        test_f1.append(f1[-1])
                    else:
                        one_val_loss, _ = self.test(model=self.model, optim_it=self.cl[i+1],
                                                    x_train=self.x_train, y_train=self.y_train,
                                                    x_test=self.x_test, y_test=self.y_test)
                    val_loss.append(one_val_loss[-1])

                val_loss = np.mean(val_loss)
                if self.REUPLOAD:
                    test_acc = np.mean(test_acc)
                    test_f1 = np.mean(test_f1)

                if self.args.record:
                    if self.REUPLOAD:
                        wandb.log({"Mean Train Loss": np.mean(all_loss), "Final Train Loss": all_loss[-1], "Mean Test Loss": val_loss,
                                "CL": self.cl[i], "Test ACC": test_acc, "Test F1": test_f1})
                    else:
                        wandb.log({"Mean Train Loss": np.mean(all_loss), "Final Train Loss": all_loss[-1], "Mean Test Loss": val_loss,
                                "CL": self.cl[i]})
                    wandb.watch(self.model, log='all', log_freq=10)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    stop = False
                    torch.save(best_model, best_model_path)
                    print('Saving best model checkpoint to', best_model_path)

                if self.REUPLOAD:
                    print("Epoch: {0:<5} CL: {1:<3} | Train Loss: {2:<8.3f} \
                          Test Loss: {3:<8.3f} Test ACC: {2:<8.3f} Test F1: {3:<8.3f}".format( \
                          n, self.cl[i], all_loss[-1], val_loss, test_acc, test_f1))
                else:
                    print("Epoch: {0:<5} CL: {1:<3} | Train Loss: {2:<8.3f} Test Loss: {3:<8.3f}".format(n, self.cl[i], all_loss[-1], val_loss))

            if (stop == True) or (i+2 >= len(self.cl)):
                print(f'[Stopping] best loss: {best_loss:.3f} CL:{i}')
                break
            i = i + 1

        torch.save(best_model, best_model_path)
        print('Saving best model to', best_model_path)
        # wandb saving best model
        if self.args.record:
            wandb.save(best_model_path)
            artifact = wandb.Artifact(setting, type='model')
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
        return best_loss, best_model


    def test(self, model, optim_it, x_train=None, y_train=None, x_test=None,
             y_test=None, verbose=False, save_traj=False):
        start_time = time.time()
        model.eval()
        ### Optimizee dependent ###
        if self.REUPLOAD: optimizee = get_optimizee(self.dimension, op_type='REUPLOAD',
                                                    x=x_train, y=y_train)
        elif self.DRESSED: optimizee = get_optimizee(self.dimension, op_type='DRESSED',
                                                     q_delta=self.q_delta, model=self.dressed_model)
        else: optimizee = get_optimizee(self.dimension)
        ### Optimizee dependent ###
        target = self.cost_func
        n_params = 0
        for name, p in optimizee.all_named_parameters():
            n_params += int(np.prod(p.size()))
        hidden_states = [Variable(torch.zeros(n_params, model.hidden_sz)) for _ in range(2)]
        cell_states = [Variable(torch.zeros(n_params, model.hidden_sz)) for _ in range(2)]
        all_acc_train = []
        all_losses_ever, all_acc_ever, all_f1_ever = [], [], []
        wall_time = []
        all_losses = None
        traj = []
        for n in range(1, optim_it + 1):
            if self.DRESSED:
                loss, train_acc = optimizee(target, 'train')
                all_acc_train.append(train_acc)
            else:
                loss = optimizee(target)

            ### Optimizee dependent ###
            if self.REUPLOAD:
                y_predict_train, _ = evaluate_REUPLOAD(optimizee.all_named_parameters()[0][1].detach().cpu(),
                                                       x_train, target, self.qnode)
                y_predict_test , _ = evaluate_REUPLOAD(optimizee.all_named_parameters()[0][1].detach().cpu(),
                                                       x_test , target, self.qnode)
                metrics_train = classification_metrics(y_train, y_predict_train)
                metrics_test  = classification_metrics(y_test , y_predict_test)
                all_acc_ever.append(metrics_test['accuracy'])
                all_f1_ever.append(metrics_test['f1'])
                if verbose:
                    print("Epoch: {0:<5} | Test Loss: {3:<8.3f} Train ACC: {2:<8.3f} Test ACC: {3:<8.3f}".format(\
                           n, loss.data.cpu().numpy(), metrics_test['accuracy'], metrics_test['accuracy']), end="\r", flush=True)
            if self.DRESSED:
                validation_loss, validation_acc = optimizee(target, 'val')
                all_acc_ever.append(validation_acc)
                if verbose:
                    print("Epoch: {0:<5} | Train Loss: {3:<8.3f} Train ACC: {2:<8.3f} Validation ACC: {3:<8.3f}".format(\
                           n, loss.data.cpu().numpy(), train_acc, validation_acc), end="\r", flush=True)

            if verbose:
                print("Epoch: {0:<5} | Test Loss: {1:<8.4f}".format(\
                        n, loss.data.cpu().numpy()), end="\r", flush=True)
            ### Optimizee dependent ###

            if all_losses is None:
                all_losses = loss
            else:
                all_losses += loss

            all_losses_ever.append(loss.data.cpu().numpy())
            loss.backward(retain_graph=False)

            ### Optimizee dependent ###
            if self.DRESSED:
                optimizee.theta = optimizee.model.fc.q_params.clone()
                optimizee.theta.grad = optimizee.model.fc.q_params.grad
            ### Optimizee dependent ###

            offset = 0
            result_params = {}
            hidden_states2 = [Variable(torch.zeros(n_params, model.hidden_sz)) for _ in range(2)]
            cell_states2 = [Variable(torch.zeros(n_params, model.hidden_sz)) for _ in range(2)]

            for name, p in optimizee.all_named_parameters():
                if save_traj:
                    traj.append(p.detach().clone())
                if p.grad is not None:
                    cur_sz = int(np.prod(p.size()))
                    gradients = detach_var(p.grad.view(cur_sz, 1))
                    updates, new_hidden, new_cell = model(
                        p, gradients,
                        [h[offset:offset+cur_sz] for h in hidden_states],
                        [c[offset:offset+cur_sz] for c in cell_states]
                    )
                    for i in range(len(new_hidden)):
                        hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                        cell_states2[i][offset:offset+cur_sz] = new_cell[i]
                    result_params[name] = p + updates
                    result_params[name].retain_grad()

                    offset += cur_sz
                else:
                    result_params[name] = p.clone()

            all_losses = None

            elapsed_time = time.time() - start_time
            wall_time.append(elapsed_time)
            ### Optimizee dependent ###
            if self.REUPLOAD: optimizee = get_optimizee(self.dimension, op_type='REUPLOAD',
                                                        x=x_train, y=y_train)
            elif self.DRESSED: optimizee = get_optimizee(self.dimension, op_type='DRESSED',
                                                         q_delta=self.q_delta, model=self.dressed_model)
            else: optimizee = get_optimizee(self.dimension)
            ### Optimizee dependent ###

            optimizee.load_state_dict(result_params, strict=False)
            optimizee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
        ### Optimizee dependent ###
        if self.REUPLOAD: return all_losses_ever, all_acc_ever, all_f1_ever
        elif self.DRESSED: return all_losses_ever, all_acc_ever, all_acc_train
        ### Optimizee dependent ###
        else:
            if save_traj: return all_losses_ever, traj, wall_time
            else: return all_losses_ever, wall_time

def evaluate_REUPLOAD(p, x, target, qnode):
    fidelity_values = []
    dm_labels = [target.density_matrix(s) for s in target.state_labels]
    predicted = []
    for i in range(len(x)):
        fidelities = [qnode([p, x[i], dm]) for dm in dm_labels]
        best_fidel = np.argmax(fidelities)

        predicted.append(best_fidel)
        fidelity_values.append(fidelities)
    return np.array(predicted), np.array(fidelity_values)
