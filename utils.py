import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from collections import defaultdict
import pandas as pd

def get_data_one_series(X, Y, n_steps, n_out, device):
    N = X.shape[0]
    steps = n_steps + n_out
    X_out = np.zeros((N // steps , n_steps, X.shape[1]), dtype=np.float64)
    Y_out = np.zeros((N // steps , n_out, Y.shape[1]), dtype=np.float64)
    x_out = 0
    y_out = 0
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    X = (X.copy() - X_mean) / X_std
    for i in range(0, N-steps+1, steps):
        X_out[x_out : x_out + n_steps, :] = X[i: i + n_steps]
        Y_out[y_out : y_out + n_out, :] = Y[i + n_steps: i + steps]
        x_out += n_steps
        y_out += n_out
    
    X_out = torch.tensor(X_out, device = device).reshape(-1, n_steps)
    Y_tensor = torch.tensor((Y_out.copy() - Y_mean) / Y_std, device=device).reshape(-1, n_out)
    return X_out, Y_tensor, Y_out.reshape(-1, n_out), X_mean, X_std, Y_mean, Y_std

def construct_latex_table(in_path, out_path):

    latex_table = defaultdict(list)
    rows = []
    columns = ["optimizer", "model", "lr", "total time", "func_calls",\
                    "n_iter", "maxit", "no. params $<$ 1e-6"]
    for root, dirs, files in os.walk(in_path, topdown=False):  
        for name in files:
            
            row = {}
            with open(os.path.join(root, name), 'r') as f:
                for i in range(16):
                    line = f.readline().strip("\n").split(':')
                    if len(line) < 2:
                        break
                    key, value = line
                    key = key.strip()
                    value = value.strip()
                    if key == "Optimizer name":
                        row['optimizer'] = value
                    elif key == "Model name":
                        row['model'] = value
                    elif key == "Learning rate":
                        row['lr'] = float(value)
                    elif key == 'The number of input features':
                        row['n_in'] = value
                    elif key == "Total number of function evaluations":
                        row['func_calls'] = int(value)
                    elif key == "Max number of iterations":
                        row['maxit'] = int(value)
                    elif key == "Total number of iterations":
                        row['n_iter'] = int(value)
                    elif key == "Percentage of parameters < 1e-6":
                        row['no. params $<$ 1e-6'] = "{:.3f}\%".format(float(value[:-1]))
                    elif key == "Total training time":
                        row['total time'] = float("{:.3f}".format(float(value)))
                    elif key == "Beta type":
                        row['beta type'] = value
            if row['maxit'] == 1:
                continue
            if 'beta type' in row:
                beta_type = row['beta type']
                if beta_type == "FR_PR":
                    beta_type = "FR\\_PR"
                row['optimizer'] = f"\\texttt{{{row['optimizer']}\\_{beta_type}}}"
                del row['beta type']
            row['model'] = "".join(row['model'].split('_'))
            if row['model'] == "MLPMultistep":
                row['model'] += row['n_in']
            del row['n_in']
            if 'func_calls' not in row:
                row['func_calls'] = "N/A"
            if 'n_iter' not in row:
                row['n_iter'] = "N/A"
            r = []
            for col in columns:
                r.append(row[col])
            rows.append(r)
            
    rows.sort()
    # for row in rows:
    #     for col, col_name in enumerate(columns):
    #         latex_table.appen

    table = "\hfill\n" +\
            "\\begin{center}\n" +\
            "\\hspace*{-3cm}\n" +\
            "\\begin{tabular}{||c | c | c | c | c | c | c | c ||}\n" +\
            "\t\\hline\n" +\
            "\toptimizer & model & lr & total time & \\texttt{func\_calls} " +\
            "& \\texttt{n\_iter} & maxit & no. params < 1e-6 \\\\ [0.5ex]\n" +\
            "\t\\hline\\hline\n"
    
    df = pd.DataFrame(rows, columns=columns)
    for i in range(df.shape[0]):
        for col in range(len(columns)):
            table += "\t" if col == 0 else " & " 
            table += str(df.iloc[i, col])
        table += "\\\\"
        if i + 1 == df.shape[0]:
            table += " [1ex]"
        table += "\n\t\\hline\n"
    table += "\\end{tabular}\n" +\
            "\\end{center}\n"
    with open(out_path, 'w') as f:
        f.write(table)

class Visualizer:
    def __init__(self, figure_path):
        self.figure_path = figure_path

    def graph_loss(self, losses, title, file_path):
        fig = plt.figure()
        plt.plot(np.arange(len(losses)), losses)
        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        fig.savefig(file_path, dpi=fig.dpi)
        plt.close()

    '''
    Args:
        model       : (nn.Module)
        data_test   : input of the model of n elements(tensor)
        y_test      : 1D Real data of n elements (np.array)
        Y_mean      : mean of y_train (np.array)
        Y_std       : std of y_train (np.array)
        model_name  : name of model to use in the graph's title
        method_name : name of the optimizer to use in the graph's title
    '''

    def graph(  self, 
                model,
                data_test,
                y_test,
                Y_mean,
                Y_std,
                model_name,
                method_name,
                file_name):

        with torch.no_grad():
            y_predict = model(data_test).detach().numpy().reshape(-1)
        y_predict = y_predict * Y_std + Y_mean
        fig = plt.figure(figsize=(14, 7), dpi=100)
        x = np.arange(len(y_test))
        plt.plot(x,y_test,  label='Real')
        plt.plot(x,y_predict, label='Predicted')
        plt.xlabel('Days')
        plt.ylabel('Close')
        plt.title('{} with {} Prediction'.format(model_name, method_name))
        plt.legend()
        folder = os.path.join(self.figure_path , method_name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = os.path.join(folder, f"{file_name}.png")
        fig.savefig(file_path, dpi=fig.dpi)
        plt.close()

    '''
    Args:
        model       : (nn.Module)
        data_test   : input of the model of n elements(tensor)
        y_test      : 1D Real data of n elements (np.array)
        Y_mean      : mean of y_train (np.array)
        Y_std       : std of y_train (np.array)
        model_name  : name of model to use in the graph's title
        method_name : name of the optimizer to use in the graph's title
        n_in        : number of input features
        n_out       : number of output features
    '''

    def graph_multistep(    self,
                            model,
                            data_test,
                            y_test,
                            Y_mean,
                            Y_std,
                            number_of_test_samples,
                            model_name,
                            method_name,
                            n_in,
                            n_out,
                            file_name,
                            device):

        fig = plt.figure()
        with torch.no_grad():
            y_predict = model(data_test).detach().numpy() * Y_mean + Y_std
        x_range = np.arange(number_of_test_samples).reshape(number_of_test_samples, 1)
        x_data, _, x_test, _, _, _, _ = get_data_one_series(x_range, x_range, n_in, n_out, device)
        x_plot = x_test.reshape(-1)
        y_predict_plot = y_predict.reshape(-1)
        y_test_plot = y_test.reshape(-1)
        sns.lineplot(y=y_predict_plot, x=x_plot, label="Predicted")
        sns.lineplot(y=y_test_plot, x=x_plot, label="Real")
        plt.title("{} with {}".format(model_name, method_name))
        plt.legend()
        # plt.show()
        folder = os.path.join(self.figure_path , method_name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = os.path.join(folder, f"{file_name}.png")
        fig.savefig(file_path, dpi=fig.dpi)
        plt.close()

    # https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html
    def graph_simple_func(self, model, data_in, optim_name):
        x = np.linspace(0, 5, 50)
        y = np.linspace(0, 5, 40)

        X, Y = np.meshgrid(x, y)
        dim = X.shape
        data = np.array(list(zip(X.reshape(-1), Y.reshape(-1))))
        data = torch.tensor(data, device = data_in.device)
        with torch.no_grad():
            Z = model(data).detach().numpy().reshape(dim)
        X = X.reshape(dim)
        Y = Y.reshape(dim)
        fig = plt.figure()
        cs = plt.contourf(X,Y,Z)
        cb = plt.colorbar(cs, orientation = 'vertical')
        plt.plot(*data_in.detach().numpy().reshape(-1), label='x', color='green', marker='o', linestyle='dashed',
            linewidth=2, markersize=12)
        plt.title(r'$e^{{(x_1 + x_2)}}$ using {optim_name}'.format(optim_name=optim_name))
        plt.legend()
        fig.savefig(f"{self.figure_path}/SimpleFunc/{model.__name__}_{optim_name}.png", dpi=fig.dpi)
        plt.close()

    # https://towardsdatascience.com/simple-little-tables-with-matplotlib-9780ef5d0bc4
    def save_table(self, title, data, image_name):
        # Pop the headers from the data array
        column_headers = data.pop(0)
        # Table data needs to be non-numeric text. Format the data
        # while I'm at it.
        cell_text = []
        for row in data:
            cell_text.append([f'{x/1000:1.1f}' for x in row])
        # Get some lists of color specs for row and column headers
        ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
        # Create the figure. Setting a small pad on tight_layout
        # seems to better regulate white space. Sometimes experimenting
        # with an explicit figsize here can produce better outcome.
        plt.figure(linewidth=2,
                edgecolor=fig_border,
                tight_layout={'pad':1},
                #figsize=(5,3)
                )
        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=cell_text,
                            rowColours=rcolors,
                            rowLoc='right',
                            colColours=ccolors,
                            colLabels=column_headers,
                            loc='center')
        # Scaling is the only influence we have over top and bottom cell padding.
        # Make the rows taller (i.e., make cell y scale larger).
        the_table.scale(1, 1.5)
        # Hide axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Hide axes border
        plt.box(on=None)
        # Add title
        plt.suptitle(title)

        # Without plt.draw() here, the title will center on the axes and not the figure.
        plt.draw()
        # Create image. plt.savefig ignores figure edge and face colors, so map them.
        fig = plt.gcf()
        fig.savefig(os.path.join(self.figure_path, image_name),
                    #bbox='tight',
                    edgecolor=fig.get_edgecolor(),
                    facecolor=fig.get_facecolor(),
                    dpi=150
                    )
        plt.close()