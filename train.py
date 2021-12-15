from models import *
from torch.optim import SGD, Adam, LBFGS
from ConjugateGradientDescent import NonlinearCG
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from utils import *

class Train:

    def __init__(   self,
                    loss_path,
                    profile_path,
                    data_path,
                    figure_path):

        self.loss_path = loss_path
        self.profile_path = profile_path
        self.data_path = data_path
        self.print_log = False
        self.print_info = False
        torch.set_default_dtype(torch.float64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer = Visualizer(figure_path)
        self.load_data()
        # print(globals())
        self.profile()

    def load_data(self):
        uniqlo_training_df = pd.read_csv(self.data_path)
        X_uniqlo_train = uniqlo_training_df[['Low', 'High']]
        Y_uniqlo_train = uniqlo_training_df['Close']
        X_train, X_test, Y_train, Y_test = train_test_split(    X_uniqlo_train,
                                                                Y_uniqlo_train,
                                                                test_size=0.3,
                                                                shuffle=False)
        X_train_np = X_train[['Low', 'High']].to_numpy(dtype=np.float64)
        Y_train_np = Y_train.to_numpy(dtype=np.float64)
        X_mean = np.mean(X_train_np, axis=0)
        X_std = np.std(X_train_np, axis=0)
        Y_mean = np.mean(Y_train_np, axis=0)
        Y_std = np.std(Y_train_np, axis=0)

        X_test_np = X_test[['Low', 'High']].to_numpy(dtype=np.float64)
        Y_test_np = Y_test.to_numpy(dtype=np.float64)
        data_test = torch.tensor(X_test_np, device = self.device)

        normalized_y_train = (Y_train_np - Y_mean) / Y_std
        data_train = torch.tensor(X_train_np, device = self.device)
        target_train = torch.tensor(normalized_y_train, device = self.device).reshape(-1,1)
        self.data = {
            'MLP_2': {  'data_train': data_train,
                        'data_train_mean': X_mean,
                        'data_train_std': X_std,
                        'data_test': data_test},
            'MLP_Large_2': {  'data_train': data_train,
                        'data_train_mean': X_mean,
                        'data_train_std': X_std,
                        'data_test': data_test}
        }

        self.target = {
            'MLP_2': {  'target_train': target_train,
                        'target_train_size': X_train_np.shape[0],
                        'target_train_transformed_size': X_train_np.shape[0],
                        'target_test_mean': Y_mean,
                        'target_test_std': Y_std,
                        'target_test': Y_test_np,
                        'target_test_size': Y_test_np.shape[0],
                        'target_test_transformed_size': Y_test_np.shape[0]
                    },
            'MLP_Large_2': {  'target_train': target_train,
                        'target_train_size': X_train_np.shape[0],
                        'target_train_transformed_size': X_train_np.shape[0],
                        'target_test_mean': Y_mean,
                        'target_test_std': Y_std,
                        'target_test': Y_test_np,
                        'target_test_size': Y_test_np.shape[0],
                        'target_test_transformed_size': Y_test_np.shape[0]
                        }
        }

        for n_in in [3,5,10,20]:
            n_out = 2
            data_train_multi, target_train_multi, target_train_multi_np, data_train_multi_mean, \
                    data_train_multi_std, target_train_multi_mean, target_train_multi_std\
                    = get_data_one_series(Y_train_np.reshape(-1,1), Y_train_np.reshape(-1,1),\
                        n_steps=n_in, n_out=n_out, device=self.device)
            data_test_multi, target_test_multi, target_test_multi_np, _ , _, _, _\
                    = get_data_one_series(Y_test_np.reshape(-1,1), Y_test_np.reshape(-1,1),\
                        n_steps=n_in, n_out=n_out, device=self.device)
            self.data[f'MLP_Multistep_{n_in}'] = {'data_train': data_train_multi,
                                            'data_test': data_test_multi,
                                            
            }
            self.target[f'MLP_Multistep_{n_in}'] = {
                'target_train': target_train_multi,
                'target_train_size': X_train_np.shape[0],
                'target_train_transformed_size': data_train_multi.shape[0],
                'target_test_mean': target_train_multi_mean,
                'target_test_std': target_train_multi_std,
                'target_test': target_test_multi_np,
                'target_test_size': Y_test_np.shape[0],
                'target_test_transformed_size': data_test_multi.shape[0]
            }

    def get_network_info(self, model):
        info = {'zero_percentage': 0,
                'total_params': 0}
        EPS_e9 = 1e-9
        EPS_e7 = 1e-7
        EPS_e6 = 1e-6

        total_params = 0
        zeros_e9 = 0
        zeros_e7 = 0
        zeros_e6 = 0

        for param in model.parameters():
            total_params += param.numel()
            zeros_e9 += (param < EPS_e9).sum()
            zeros_e7 += (param < EPS_e7).sum()
            zeros_e6 += (param < EPS_e6).sum()

        info['zero_percentage_e9'] = zeros_e9 / total_params * 100
        info['zero_percentage_e7'] = zeros_e7 / total_params * 100
        info['zero_percentage_e6'] = zeros_e6 / total_params * 100
        info['total_params'] = total_params
        return info

    def train_closure(  self,
                        maxit,
                        model,
                        optimizer,
                        data_train,
                        target_train,
                        print_freq = 50,
                        prevent_loss_explosion=True):
        it = [0]
        last_loss = 0
        current_loss = 0
        losses = []
        start = time.time()
        loss_fn = nn.MSELoss()
        while it[0] < maxit:
            def closure():
                optimizer.zero_grad()
                out = model(data_train)
                loss = loss_fn(out, target_train)
                nonlocal current_loss
                nonlocal last_loss
                last_loss = current_loss
                current_loss = loss.item()
                loss.backward()
                it[0] += 1
                return loss
            orig_loss = optimizer.step(closure)
            if isinstance(orig_loss, list):
                losses.extend([loss for loss in orig_loss])
            else:
                losses.append(orig_loss.item())

            if it[0] > 2 and (abs(current_loss - last_loss) < 1e-9 or \
                            (prevent_loss_explosion and abs(current_loss - last_loss) > 1e6)):
                break
        end = time.time()
        network_info = self.get_network_info(model)
        network_info['total_time'] = end-start
        network_info['losses'] = losses
        network_info['model'] = model
        network_info['optimizer'] = optimizer
        network_info['maxit'] = maxit
        self.write_profiler(network_info)
    
    def train(  self,
                maxit,
                model,
                optimizer,
                data_train,
                target_train,
                print_freq = 1,
                prevent_loss_explosion=True):

        it = [0]
        last_loss = 0
        current_loss = 0
        start = time.time()
        losses = []
        loss_fn = nn.MSELoss()
        while it[0] < maxit:
            optimizer.zero_grad()
            out = model(data_train)
            loss = loss_fn(out, target_train)
            loss.backward()
            optimizer.step()
            last_loss = current_loss
            current_loss = loss.item()
            losses.append(current_loss)
            it[0] += 1
            if it[0] > 2 and (abs(current_loss - last_loss) < 1e-9 or \
                            (prevent_loss_explosion and abs(current_loss - last_loss) > 1e6)):
                break
        end = time.time()
        network_info = self.get_network_info(model)
        network_info['total_time'] = end-start
        network_info['losses'] = losses
        network_info['model'] = model
        network_info['optimizer'] = optimizer
        network_info['maxit'] = maxit
        self.write_profiler(network_info)

    def train_wrapper(  self,
                        model_name,
                        optim_name,
                        model,
                        optimizer,
                        data_train,
                        target_train
                        ):

        if optim_name == "SGD":
            self.train(3000, model, optimizer, data_train, target_train, print_freq=1)
        elif optim_name == "Adam":
            self.train(3000, model, optimizer, data_train, target_train, print_freq=50)
        elif optim_name == "LBFGS":
            self.train_closure(3000, model, optimizer, data_train, target_train, print_freq = 50)
        else:
            # print(optim_name)
            assert optim_name == "NonlinearCG"
            self.train_closure(1, model, optimizer, data_train, target_train, print_freq = 50)
            self.train_closure(3000, model, optimizer, data_train, target_train, print_freq = 50)

    def write_profiler(self, network_info):

        optimizer = network_info['optimizer']
        model = network_info['model']
        model_name = model.__class__.__name__
        optimizer_name = optimizer.__class__.__name__
        maxit = network_info['maxit']
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        zeros_e9 = network_info['zero_percentage_e9']
        zeros_e7 = network_info['zero_percentage_e7']
        zeros_e6 = network_info['zero_percentage_e6']
        total_time = network_info['total_time']
        losses = network_info['losses']
        total_params = network_info['total_params']
        n_in = model.n_in
        n_out = model.n_out
        number_of_train_samples = self.target[f"{model_name}_{n_in}"]['target_train_transformed_size']
        number_of_test_samples = self.target[f"{model_name}_{n_in}"]['target_test_transformed_size']

        file_name = f"{model_name}_{optimizer_name}_{lr}_{n_in}"
        line =  f"Model name                            : {model_name}\n"
        line += f"The number of input features          : {n_in}\n"
        line += f"The number of output features         : {n_out}\n"
        line += f"Optimizer name                        : {optimizer_name}\n"
        line += f"Learning rate                         : {lr}\n"

        if optimizer_name == "NonlinearCG":
            beta_type = optimizer.state_dict()['param_groups'][0]['beta_type']
            line += f"Beta type                             : {beta_type}\n"
            file_name += f"_{maxit}_{beta_type}"

        folder = os.path.join(self.profile_path, optimizer_name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = os.path.join(folder, file_name+".txt")
        
        with open(file_path, 'w') as f:
            
            if optimizer_name in ["LBFGS", "NonlinearCG"]:
                func_evals = optimizer.state_dict()['state'][0]['func_evals']
                n_iter = optimizer.state_dict()['state'][0]['n_iter']
                line += f"Total number of function evaluations  : {func_evals}\n"
                line += f"Total number of iterations            : {n_iter}\n"

            line += f"Max number of iterations              : {maxit}\n"
            line += f"Number of samples in training data    : {number_of_train_samples}\n"
            line += f"Number of samples in tests data       : {number_of_test_samples}\n"
            line += f"Total training time                   : {total_time}\n"
            line += f"Total number of parameters            : {total_params}\n"
            line += f"Percentage of parameters < 1e-9       : {zeros_e9}%\n"
            line += f"Percentage of parameters < 1e-7       : {zeros_e7}%\n"
            line += f"Percentage of parameters < 1e-6       : {zeros_e6}%\n"
            line += "\n"
            line += "=============================================\n"
            line += "=============================================\n"
            line += "===================Losses====================\n"
            line += "=============================================\n"
            line += "=============================================\n"
            line += "\n"
            line += "\n"
            f.write(line)
            line = ""
            last_loss = None
            for it, loss in enumerate(losses):
                line += f"Loss at iteration [{it+1}]: {loss}\n"
                if it > 1 and loss > last_loss:
                    line += f"***** Warning: Loss has increased *****\n"
                last_loss = loss
            f.write(line)

        model_name_data = f"{model_name}_{n_in}"
        if model_name == "MLP_Multistep":
            self.visualizer.graph_multistep(    model,
                                                self.data[model_name_data]['data_test'],
                                                self.target[model_name_data]['target_test'],
                                                self.target[model_name_data]['target_test_mean'],
                                                self.target[model_name_data]['target_test_std'],
                                                self.target[model_name_data]['target_test_size'],
                                                f"{model_name} with {n_in} features",
                                                optimizer_name,
                                                n_in,
                                                n_out,
                                                file_name,
                                                self.device)
        else:
            self.visualizer.graph(  model,
                                    self.data[model_name_data]['data_test'],
                                    self.target[model_name_data]['target_test'],
                                    self.target[model_name_data]['target_test_mean'],
                                    self.target[model_name_data]['target_test_std'],
                                    f"{model_name} with {n_in} features",
                                    optimizer_name,
                                    file_name)
        title = f"Loss of {model_name} using {optimizer_name} with lr={lr}, n_in={n_in}, maxit={maxit}"
        folder = os.path.join(self.loss_path, f"{optimizer_name}")
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = os.path.join(folder, file_name+".png")
        self.visualizer.graph_loss(losses, title, file_path)

    def profile(self):
        sgd_lrs = [0.2, 0.3, 0.4, 0.01, 0.001, 0.0001]
        lrs = [0.1, 0.01, 0.001, 0.0001]

        lrs_dict = {
            'SGD': sgd_lrs,
            'LBFGS': lrs,
            'Adam': lrs,
            'NonlinearCG': lrs
        }

        in_dict = {
            "MLP": [2],
            "MLP_Large": [2],
            "MLP_Multistep": [3,5,10,20]
        }

        opt_params = {
            'SGD': [{'momentum' : 0.5}],
            'LBFGS': [{'line_search_fn' : 'strong_wolfe'}],
            'Adam': [{}],
            'NonlinearCG': [{'line_search_fn' : 'strong_wolfe', 'beta_type': "FR_PR"},
                            {'line_search_fn' : 'strong_wolfe', 'beta_type': "HS"}]
        }

        model_params = {
            'MLP': {    'data_train_mean':self.data['MLP_2']['data_train_mean'],
                        'data_train_std':self.data['MLP_2']['data_train_std'],
                        'device': self.device },
            'MLP_Large': {  'data_train_mean':self.data['MLP_Large_2']['data_train_mean'],
                            'data_train_std':self.data['MLP_Large_2']['data_train_std'],
                            'device': self.device},
            'MLP_Multistep': {}
        }

        for model_name in ["MLP", "MLP_Large", "MLP_Multistep"]:
        # for model_name in ["MLP_Multistep"]:
            # for optim_name in ["LBFGS", "SGD", "Adam", "NonlinearCG"]:
            for optim_name in ["NonlinearCG"]:
                for lr in lrs_dict[optim_name]:
                    for n_in in in_dict[model_name]:
                        for opt_param in opt_params[optim_name]:
                            print(f"Training for model: {model_name}, optimizer: {optim_name}, "+
                                    f"learning rate: {lr}, number of features: {n_in}")
                            data_train = self.data[f"{model_name}_{n_in}"]['data_train']
                            target_train = self.target[f"{model_name}_{n_in}"]['target_train']
                            
                            # print(f"n_in: {n_in}, model: {model_name}, optim: {optim_name}")
                            # print(model_params[model_name], globals()[model_name], type(MLP), type(globals()[model_name]))
                            model = globals()[model_name](  n_in,\
                                                            **model_params[model_name])
                            model.requires_grad_(True)
                            optimizer = globals()[optim_name](  model.parameters(),
                                                                            lr=lr,
                                                                            **opt_param)
                            self.train_wrapper( model_name,
                                                optim_name,
                                                model,
                                                optimizer,
                                                data_train,
                                                target_train)


    
