from train import Train
from utils import construct_latex_table
import os

if __name__ == "__main__":
    train = Train(  "./logs/test/loss_log",
                    "./logs/test/profile_log",
                    "./data/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv",
                    "./logs/test/figures")
    # profile_path = "./logs/profile_log"
    # latex_path = "./logs/latex_table"
    # optimizers = ["LBFGS", "NonlinearCG", "Adam", "SGD"]
    # for opt in optimizers:
    #     in_path = os.path.join(profile_path, opt)
    #     out_path = os.path.join(latex_path, opt+".txt")
    #     if not os.path.exists(latex_path):
    #         os.mkdir(latex_path)
    #     construct_latex_table(in_path, out_path)
