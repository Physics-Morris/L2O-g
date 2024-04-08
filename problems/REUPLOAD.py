import torch
import pennylane as qml
from utils.meta_module import *
import numpy as np


class REUPLOAD:
    def __init__(self, device):
        self.device = device

    def setup_problem(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_loss_function(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_optimizee(self):
        raise NotImplementedError("Subclasses should implement this method.")

class REUPLOAD_problem(REUPLOAD):
    def __init__(self, func_type='circle', qubits=1, layers=3, splits=[.7, .2, .1], n_data=5000, batch_size=32, seed=0):
        self.func_type = func_type
        self.qubits = qubits
        self.layers = layers
        self.batch_size = batch_size
        self.seed = seed
        self.state_labels = torch.tensor([[[1], [0]], [[0], [1]]], requires_grad=False)
        # self.splits = [np.floor(splits[0]*n_data).astype(int), np.floor(splits[1]*n_data).astype(int), 
        #                n_data-(np.floor(splits[0]*n_data).astype(int)+np.floor(splits[1]*n_data).astype(int))]
        # from data reupload paper
        self.splits = [200, 200, 4000]
        self.n_data = n_data
        device = qml.device("lightning.qubit", wires=qubits)
        super().__init__(device)
        self.setup_problem()
        self.define_qnode()

    # Make a dataset of points inside and outside of a circle
    def circle(self, samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
        """
        Generates a dataset of points with 1/0 labels inside a given radius.

        Args:
            samples (int): number of samples to generate
            center (tuple): center of the circle
            radius (float: radius of the circle

        Returns:
            Xvals (array[tuple]): coordinates of points
            yvals (array[int]): classification labels
        """
        Xvals, yvals = [], []

        for _ in range(samples):
            x = 2 * (np.random.rand(2)) - 1
            y = 0
            if np.linalg.norm(x - center) < radius:
                y = 1
            Xvals.append(x)
            yvals.append(y)
        return torch.tensor(Xvals, requires_grad=False), torch.tensor(yvals, requires_grad=False)

    def density_matrix(state):
        """Calculates the density matrix representation of a state using PyTorch.

        Args:
            state (Tensor): a tensor representing a quantum state vector, must be a complex tensor

        Returns:
            Tensor: a tensor representing the density matrix
        """
        if not isinstance(state, torch.Tensor):
            raise ValueError("state must be a PyTorch tensor")

        state_conj_transpose = torch.conj(state).T
        dm = state * state_conj_transpose

        return dm

    def setup_problem(self):
        np.random.seed(self.seed)
        print('Preparing dataset', self.func_type, 'total data point', sum(self.splits))
        if self.func_type == 'circle':
            x_train, self.y_train = self.circle(self.splits[0])
            x_val  , self.y_val   = self.circle(self.splits[1])
            x_test , self.y_test  = self.circle(self.splits[2])
        else: print('func_type ', self.func_type, ' not implemented')
        self.x_train = torch.cat((x_train, torch.zeros(x_train.shape[0], 1, requires_grad=False)), dim=1)
        self.x_val   = torch.cat((x_val,   torch.zeros(x_val.shape[0]  , 1, requires_grad=False)), dim=1)
        self.x_test  = torch.cat((x_test,  torch.zeros(x_test.shape[0] , 1, requires_grad=False)), dim=1)

    def ansatz(self, params):
        """A variational quantum circuit representing the Universal classifier.

        Args:
            params (array[float]): array of parameters
            x (array[float]): single input vector
            y (array[float]): single output state density matrix

        Returns:
            float: fidelity between output state and input
        """
        [ps, x, y] = params
        ps = ps.reshape((self.layers, 3))
        for p in ps:
            qml.Rot(*x, wires=0)
            qml.Rot(*p, wires=0)
        return qml.expval(qml.Hermitian(y, wires=[0]))

    def test(self, params, x, y, state_labels=None):
        """
        Tests on a given set of data.

        Args:
            params (array[float]): array of parameters
            x (array[float]): 2-d array of input vectors
            y (array[float]): 1-d array of targets
            state_labels (array[float]): 1-d array of state representations for labels

        Returns:
            predicted (array([int]): predicted labels for test data
            output_states (array[float]): output quantum states from the circuit
        """
        fidelity_values = []
        dm_labels = [self.density_matrix(s) for s in state_labels]
        predicted = []

        for i in range(len(x)):
            fidel_function = lambda y: self.qnode([params, x[i], y])
            fidelities = [fidel_function(dm) for dm in dm_labels]
            best_fidel = np.argmax(fidelities)

            predicted.append(best_fidel)
            fidelity_values.append(fidelities)

        return np.array(predicted), np.array(fidelity_values)

    def define_qnode(self):
        self.qnode = qml.QNode(self.ansatz, self.device, interface="torch")

    def get_metric_fn(self):
        metric_fn = lambda p: qml.metric_tensor(self.qnode, approx="diag")(p)
        return metric_fn

    def get_data(self):
        return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test

    def get_qnode(self):
        return self.qnode

    def get_loss_function(self):
        class Loss_Func:
            def __init__(self, qnode, state_labels, batch_size):
                self.qnode = qnode
                self.state_labels = state_labels
                self.batch_size = batch_size

            def density_matrix(self, state):
                """Calculates the density matrix representation of a state using PyTorch.

                Args:
                    state (Tensor): a tensor representing a quantum state vector, must be a complex tensor

                Returns:
                    Tensor: a tensor representing the density matrix
                """
                if not isinstance(state, torch.Tensor):
                    raise ValueError("state must be a PyTorch tensor")

                state_conj_transpose = torch.conj(state).T
                dm = state * state_conj_transpose

                return dm

            def iterate_minibatches(self, inputs, targets, batch_size):
                """
                A generator for batches of the input data

                Args:
                    inputs (array[float]): input data
                    targets (array[float]): targets

                Returns:
                    inputs (array[float]): one batch of input data of length `batch_size`
                    targets (array[float]): one batch of targets of length `batch_size`
                """
                for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
                    idxs = slice(start_idx, start_idx + batch_size)
                    yield inputs[idxs], targets[idxs]

            def cost(self, params, x, y, state_labels=None):
                """Cost function to be minimized.

                Args:
                    params (array[float]): array of parameters
                    x (array[float]): 2-d array of input vectors
                    y (array[float]): 1-d array of targets
                    state_labels (array[float]): array of state representations for labels

                Returns:
                    float: loss value to be minimized
                """
                # Compute prediction for each input in data batch
                loss = 0.0
                dm_labels = [self.density_matrix(s) for s in state_labels]
                for i in range(len(x)):
                    f = self.qnode([params, x[i], dm_labels[y[i]]])
                    loss = loss + (1 - f) ** 2
                return loss / len(x)

            def get_loss(self, params):
                [theta, x, y] = params
                total_loss = 0.0
                for xbatch, ybatch in self.iterate_minibatches(x, y, batch_size=self.batch_size):
                    total_loss = total_loss + self.cost(theta, xbatch, ybatch, self.state_labels)
                return total_loss

        return Loss_Func(self.qnode, self.state_labels, self.batch_size)