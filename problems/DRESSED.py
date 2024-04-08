import time
import copy
import torch
import torch.nn as nn
import pennylane as qml
from utils.meta_module import *
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from pennylane import numpy as pnp

def labels_to_filtered(filtered_labels, labels):
    """Maps CIFAR labels (0,1,2,3,4,5,6,7,8,9) to the index of filtered_labels"""
    return [filtered_labels.index(label) for label in labels]

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates. 
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)
        
def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis. 
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT  
    for i in range(0, nqubits - 1, 2): # Loop over even indices: i=0,2,...N-2  
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1,2): # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

class Quantumnet(nn.Module):
        def __init__(self, n_qubits, q_delta, max_layers, filtered_classes, qnode, device='cpu'):
            super().__init__()
            self.n_qubits = n_qubits
            self.pre_net = nn.Linear(512, n_qubits)
            self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
            self.post_net = nn.Linear(n_qubits, len(filtered_classes))
            self.device = device
            self.qnode = qnode

        def forward(self, input_features):
            pre_out = self.pre_net(input_features) 
            q_in = torch.tanh(pre_out) * np.pi / 2.0   
            
            # Apply the quantum circuit to each element of the batch, and append to q_out
            q_out = torch.Tensor(0, self.n_qubits)
            q_out = q_out.to(self.device)
            for elem in q_in:
                q_out_elem = torch.hstack(self.qnode([elem, self.q_params])).float().unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))
            return self.post_net(q_out)

class DRESSED:
    def __init__(self, device):
        self.Qdevice = device

    def setup_problem(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_loss_function(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_optimizee(self):
        raise NotImplementedError("Subclasses should implement this method.")

class DRESSED_problem(DRESSED):
    def __init__(self, task=['plane', 'car'], n_qubits=4, quantum=True, c_model='512_n', batch_size=8, 
                 q_depth=4, max_layers=15, q_delta=0.01, rng_seed=0, device='cpu'):
        self.Qdevice = qml.device("lightning.qubit", wires=n_qubits)
        self.filtered_classes = task
        self.n_qubits = n_qubits
        self.quantum = quantum
        self.classical_model = c_model
        self.batch_size = batch_size
        self.q_depth = q_depth
        self.max_layers = max_layers
        self.q_delta = q_delta
        self.rng_seed = rng_seed
        self.device = device
        super().__init__(self.Qdevice)
        self.load_dataset()
        self.define_qnode()
        self.setup_model()
        self.select_criterion()

    def select_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def setup_model(self):
        # load pre-train model
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model_hybrid = torchvision.models.resnet18(weights=weights)
        for param in model_hybrid.parameters():
            param.requires_grad = False
        if self.quantum:
            model_hybrid.fc = Quantumnet(self.n_qubits, self.q_delta, self.max_layers, self.filtered_classes, self.qnode, self.device)
        elif self.c_model == '512_n':
            model_hybrid.fc = nn.Linear(512,len(self.filtered_classes))
        elif self.c_model == '512_nq_n':
            model_hybrid.fc = nn.Sequential(nn.Linear(512, self.n_qubits),torch.nn.ReLU(),nn.Linear(self.n_qubits, len(self.filtered_classes))) 
        elif self.c_model == '551_512_n':
            model_hybrid.fc = nn.Sequential(nn.Linear(512, 512), torch.nn.ReLU(), nn.Linear(512, len(self.filtered_classes)))
        self.model = model_hybrid.to(self.device)

    def load_dataset(self):
        # Fixed pre-processing operations
        data_transforms = {
            'train': transforms.Compose([
                #transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                #transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # Normalize input channels using mean values and standard deviations of ImageNet.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # =================== begin CIFAR dataset loading ===================
        trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=data_transforms['train'])
        testset_full = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=data_transforms['val'])
        image_datasets_full={'train': trainset_full, 'val': testset_full}

        # CIFAR classes
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')

        # Get indices of samples associated to filtered_classes
        self.filtered_labels = [class_names.index(cl) for cl in self.filtered_classes]
        sub_indices={'train': [], 'val': []}
        for phase in ['train', 'val']:
            for idx, label in enumerate(image_datasets_full[phase].targets):  
                if label in self.filtered_labels:
                    sub_indices[phase].append(idx)
                    
        # Initialize sub-datasets according to filtered indices
        image_datasets = {x: torch.utils.data.Subset(image_datasets_full[x], sub_indices[x])
                        for x in ['train', 'val']}

        # Number of samples
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        # Initialize dataloader
        torch.manual_seed(self.rng_seed)
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                            batch_size=self.batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

    def ansatz(self, params):
            
            [q_in, q_weights_flat] = params

            # Reshape weights
            q_weights = q_weights_flat.reshape(self.max_layers, self.n_qubits)
            
            # Start from state |+> , unbiased w.r.t. |0> and |1>
            H_layer(self.n_qubits)
            
            # Embed features in the quantum node
            RY_layer(q_in)
        
            # Sequence of trainable variational layers
            for k in range(self.q_depth):
                entangling_layer(self.n_qubits)
                RY_layer(q_weights[k+1])

            # Expectation values in the Z basis
            exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(self.n_qubits)]
            return tuple(exp_vals)

    def define_qnode(self):
        self.qnode = qml.QNode(self.ansatz, self.Qdevice, interface="torch")

    def get_metric_fn(self):
        metric_fn = lambda p: qml.metric_tensor(self.qnode, approx="diag")(p)
        return metric_fn

    def get_qnode(self):
        return self.qnode

    def get_dressed_model(self):
        return self.model

    def get_loss_function(self):
        class Loss_Func:
            def __init__(self, model, qnode, dataloaders, criterion, batch_size, dataset_sizes, device, filtered_labels):
                self.model = model
                self.qnode = qnode
                self.dataloaders = dataloaders
                self.criterion = criterion
                self.batch_size = batch_size
                self.dataset_sizes= dataset_sizes
                self.device = device
                self.filtered_labels = filtered_labels

            def get_loss(self, model, phase):
                # Iteration loop
                total_loss = 0.0
                running_loss = 0.0
                running_corrects = 0
                n_batches = self.dataset_sizes[phase] // self.batch_size
                it = 0
                for inputs, cifar_labels in self.dataloaders[phase]:
                    batch_size_ = len(inputs)
                    inputs = inputs.to(self.device)
                    labels = torch.tensor(labels_to_filtered(self.filtered_labels, cifar_labels))
                    labels = labels.to(self.device)
                    
                    # Track/compute gradient and make an optimization step only when training
                    # with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                            
                    # Print iteration results
                    total_loss = total_loss + loss
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(preds == labels.data).item()
                    running_corrects += batch_corrects
                    it += 1
                total_loss = total_loss / self.dataset_sizes[phase]
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
                return total_loss, epoch_acc

        return Loss_Func(self.model, self.qnode, self.dataloaders, self.criterion, self.batch_size, 
                         self.dataset_sizes, self.device, self.filtered_labels)