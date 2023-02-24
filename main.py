import logging
import os
import sys
import warnings

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.tools import (
    IOStream,
    PointCloudsInDF,
    _init_,
    delete_files,
    notifi,
    plot_stats,
)
from utils.augmentation import AugmentPointCloudsInDF
from utils.train import test, train
from utils.send_telegram import send_telegram

warnings.filterwarnings("ignore")

def main(params):
    # set up folder structure
    _init_(params["exp_name"])

    # initiate IOStream
    io = IOStream("checkpoints/" + params["exp_name"] + "/run.log")
    io.cprint(params["exp_name"])

    if params["cuda"]:
        io.cprint("Using GPU")
    else:
        io.cprint("Using CPU")

    # Load datasets
    train_data_path = os.path.join(params["train_path"], str(params["num_points"]))
    train_df = params["train_df"]
    trainset = PointCloudsInDF(train_data_path, train_df)
    
    # Augment Point Cloud
    if params["augment"] == True:
        for i in range(len(classes)):
            n = params["n_augs"][i]
            train_df_i = train_df[train_df["class"] == i ]
            for i in range(n):
                aug_trainset = AugmentPointCloudsInDF(train_data_path, train_df_i,)
                
                trainset = torch.utils.data.ConcatDataset([trainset, aug_trainset])

    # Data Loader for Training
    # trainloader = DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, pin_memory=True)

    test_data_path = os.path.join(params["test_path"], str(params["num_points"]))
    test_df = params["test_df"]
    # testloader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False, pin_memory=True)

    if not params["eval"]:
        testset = PointCloudsInDF(test_data_path, test_df)
        train(params, io, trainset, testset)
        torch.cuda.empty_cache()
    else:
        testset = PointCloudsInDF(test_data_path, test_df, label=False)
        test(params, io, testset)
        
        
if __name__ == "__main__":
    # Subset Dataframes
    df = pd.read_csv(r"E:\TR3D_species\tree_metadata_training_publish_BM.csv") # read csv
    train_df = df.loc[df["train_val"] == "train"] # training
    val_df = df.loc[df["train_val"] == "val"] # validation
    test_df = pd.read_csv(r"E:\TR3D_species\test.csv")
    classes = list(range(33))
    train_prop = [100 * len(train_df[train_df["class"] == i]) / len(train_df) for i in classes]
    train_count = [len(train_df[train_df["class"] == i]) for i in classes]
    # train_weights = torch.Tensor(1/np.array(train_prop))
    train_weights = None
    n_augs = [round((max(train_count) - i) / i) for i in train_count]
    
    params = {
        "exp_name": "dgcnn_pointaugment_4096",  # experiment name
        "model": "dgcnn",  # model
        "batch_size": 24,  # batch size
        "train_path": r"E:\TR3D_species\train\datasets\cluster_fps",
        "train_df": train_df,
        "test_path": r"E:\TR3D_species\test\datasets\cluster_fps",
        "test_df": test_df,
        "augment": True, # augment
        "n_augs": n_augs, # number of augmentations
        "classes": list(range(33)),  # classes
        "n_gpus": torch.cuda.device_count(),  # number of gpus
        "epochs": 300,  # total epochs
        "optimizer_a": "adam",  # augmentor optimizer,
        "optimizer_c": "adam",  # classifier optimizer
        "lr_a": 1e-4,  # augmentor learning rate
        "lr_c": 1e-4,  # classifier learning rate
        "adaptive_lr": True,  # adaptive learning rate
        "patience": 10,  # patience
        "step_size": 20,  # step size
        "momentum": 0.9,  # sgd momentum
        "num_points": 4096,  # number of points
        "dropout": 0.5,  # dropout rate
        "emb_dims": 1024,  # dimension of embeddings
        "k": 20,  # k nearest points
        "model_path": r"D:\MurrayBrent\projects\TR3D\checkpoints\dgcnn_pointaugment_4096\models\best_model.t7",  # pretrained model path
        "cuda": True,  # use cuda
        "eval": True,  # run testing
        "train_weights": train_weights, # training weights
        "write_aug": False, # write augmented files
        "send_telegram": False,
    }
    
    mn = params["exp_name"]
    if params["send_telegrams"]:
        mn = params["exp_name"]
        token = ''
        chat_id = ''
        send_telegram(f"Starting {mn}")
    main(params)