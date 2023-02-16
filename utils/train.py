import os
import random
import warnings


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from augment.augmentor import Augmentor
from models.dgcnn import DGCNN
from common import loss_utils
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.send_telegram import send_telegram, send_photos
from utils.tools import create_comp_csv, delete_files, variable_df, write_las, plot_3d, plot_2d, make_confusion_matrix

warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (15, 15)

def train(params, io, train_set, test_set):
    # Run model
    device = torch.device("cuda" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]

    # Classifier
    if params["model"] == "dgcnn":
        classifier = DGCNN(params, len(params["classes"])).to(device).cuda()
    else:
        raise Exception("Model Not Implemented")

    # Augmentor
    augmentor = Augmentor().cuda()

    # Run in Parallel
    if params["n_gpus"] > 1:
        classifier = nn.DataParallel(
            classifier.cuda(), device_ids=list(range(0, params["n_gpus"]))
        )
        augmentor = nn.DataParallel(
            augmentor.cuda(), device_ids=list(range(0, params["n_gpus"]))
        )

    # Set up optimizers
    if params["optimizer_a"] == "sgd":
        optimizer_a = optim.SGD(
            augmentor.parameters(),
            lr=params["lr_a"],
            momentum=params["momentum"],
            weight_decay=1e-4,
        )
    elif params["optimizer_a"] == "adam":
        optimizer_a = optim.Adam(
            augmentor.parameters(), lr=params["lr_a"], betas=(0.9, 0.999), eps=1e-08
        )
    else:
        raise Exception("Optimizer Not Implemented")

    if params["optimizer_c"] == "sgd":
        optimizer_c = optim.SGD(
            classifier.parameters(),
            lr=params["lr_c"],
            momentum=params["momentum"],
            weight_decay=1e-4,
        )
    elif params["optimizer_c"] == "adam":
        optimizer_c = optim.Adam(
            classifier.parameters(), lr=params["lr_c"], betas=(0.9, 0.999), eps=1e-08
        )
    else:
        raise Exception("Optimizer Not Implemented")

    # Adaptive Learning
    if params["adaptive_lr"] is True:
        scheduler1_c = ReduceLROnPlateau(optimizer_c, "max", patience=params["patience"])
        scheduler2_c = StepLR(optimizer_c, step_size=params["step_size"], gamma=0.1)
        scheduler1_a = ReduceLROnPlateau(optimizer_a, "max", patience=params["patience"])
        scheduler2_a = StepLR(optimizer_a, step_size=params["step_size"], gamma=0.1)
        change = 0

    # Set initial best test loss
    best_test_f1 = 0.0

    # Set initial triggertimes
    triggertimes = 0
    
    if params["train_weights"] is not None:
        train_weights = params["train_weights"].to(device)
    else:
        train_weights = None
        
    test_loader = DataLoader(test_set, batch_size=params["batch_size"], shuffle=False, pin_memory=True)
    # Iterate through number of epochs
    for epoch in tqdm(
        range(params["epochs"]), desc="Model Total: ", leave=False, colour="red"
    ):
        train_loss_a = 0.0
        train_loss_c = 0.0
        count = 0
        true_pred = []
        aug_pred = []
        train_true = []
        j=0

        # Use half of training set
        trainset_idx = list(range(len(train_set)))
        trainset_idx_1 = random.sample(trainset_idx, round(len(trainset_idx) * 0.5))
        rem = len(trainset_idx_1) % params["batch_size"]
        if rem <= 3:
            trainset_idx_1 = trainset_idx_1[: len(trainset_idx_1) - rem]
        
        trainset = Subset(train_set, trainset_idx_1)
        train_loader = DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, pin_memory=True)
        
        for data, label in tqdm(
            train_loader, desc="Training Total: ", leave=False, colour="cyan"
        ):
            # Get data and label
            data, label = (data.to(device), label.to(device).squeeze())

            # Permute data into correct shape
            data = data.permute(0, 2, 1)  # adapt augmentor to fit with this permutation

            # Get batch size
            batch_size = data.size()[0]

            # # Augment
            noise = (0.02 * torch.randn(batch_size, 1024))
            noise = noise.to(device)
            
            augmentor.train()
            classifier.train()
            optimizer_a.zero_grad()  # zero gradients
            group = (data, noise)
            aug_pc = augmentor(group)

            # Classify
            out_true = classifier(data)  # classify truth
            out_aug = classifier(aug_pc)  # classify augmented
            
            # Augmentor Loss
            aug_loss = loss_utils.g_loss(label, out_true, out_aug, data, aug_pc, train_weights=None)

            # Backward + Optimizer Augmentor
            aug_loss.backward(retain_graph=True)

           
            # Classifier Loss
            optimizer_c.zero_grad()  # zero gradients
            cls_loss = loss_utils.d_loss(label, out_true, out_aug, train_weights=train_weights)
            # cls_loss = loss_utils.calc_loss(label, out_true)

            # Backward + Optimizer Classifier
            cls_loss.backward()
            optimizer_a.step()
            optimizer_c.step()

            # Update loss' and count
            train_loss_a += aug_loss.item()
            train_loss_c += cls_loss.item()
            count = batch_size
        
            
            # Append true/pred
            train_true.append(torch.argmax(label, dim=1).cpu().numpy()) # y class
            
            aug_sm = F.softmax(out_aug, dim=1)
            aug_pred.append(torch.argmax(aug_sm, dim=1).cpu().numpy()) # y_aug class
            
            true_sm = F.softmax(out_true, dim=1)
            true_pred.append(torch.argmax(true_sm, dim=1).cpu().numpy()) # y_true class
            
            if params["write_aug"]:
                if epoch + 1 in [1, 50, 100, 150, 200, 250, 300]:
                    if random.random() > 0.99:
                        aug_pc_np = aug_pc.detach().cpu().numpy()
                        true_pc_np = data.detach().cpu().numpy()
                        try:
                            write_las(aug_pc_np[1], f"checkpoints/{exp_name}/output/laz/epoch{epoch + 1}_pc{j}_aug.laz")
                            write_las(true_pc_np[1], f"checkpoints/{exp_name}/output/laz/epoch{epoch + 1}_pc{j}_true.laz")
                            j+=1
                        except:
                            j+=1
        
        # Calculate F1 scores
        
        train_true = np.concatenate(train_true)
        aug_pred = np.concatenate(aug_pred)
        true_pred = np.concatenate(true_pred)
        
        aug_f1 = f1_score(train_true, aug_pred, average="weighted")
        true_f1 = f1_score(train_true, true_pred, average="weighted")
        
        train_f1 = float(aug_f1 + true_f1) / 2
        train_f1 = float(true_f1)
        
        # Get average loss'
        train_loss_a = float(train_loss_a) / count
        train_loss_c = float(train_loss_c) / count

        # Set up Validation
        classifier.eval()
        with torch.no_grad():
            test_loss = 0.0
            count = 0
            test_pred = []
            test_true = []

            # Validation
            for data, label in tqdm(
                test_loader, desc="Validation Total: ", leave=False, colour="green"
            ):
                # Get data and label
                data, label = (data.to(device), label.to(device).squeeze())

                # Permute data into correct shape
                data = data.permute(0, 2, 1)

                # Get batch size
                batch_size = data.size()[0]

                # Run model
                output = classifier(data)

                # Calculate loss
                loss = loss_utils.calc_loss(label, output)

                # Update count and test_loss
                count += batch_size
                test_loss += loss.item()

                # Append true/pred
                
                test_true.append(torch.argmax(label, dim=1).cpu().numpy())
                
                pred_sm = F.softmax(output, dim=1)
                test_pred.append(torch.argmax(pred_sm, dim=1).cpu().numpy())


            # Calculate f1
            
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            val_f1 = f1_score(test_true, test_pred, average="weighted")

            # get average test loss
            test_loss = float(test_loss) / count
        
        # Create output dataframe
        out_dict = {"epoch": [epoch + 1],
                    "aug_loss": [train_loss_a],
                    "class_loss": [train_loss_c],
                    "train_f1": [train_f1],
                    "val_loss": [test_loss],
                    "val_f1": [val_f1]}
        out_df = pd.DataFrame.from_dict(out_dict)
        
        if epoch + 1 > 1:
            loss_f1_df = pd.read_csv(f"checkpoints/{exp_name}/loss_f1.csv")
            loss_f1_df = pd.concat([loss_f1_df, out_df])
            loss_f1_df.to_csv(f"checkpoints/{exp_name}/loss_f1.csv", index=False)
        else:
            out_df.to_csv(f"checkpoints/{exp_name}/loss_f1.csv", index=False)

        # Save Best Model
        if val_f1 > best_test_f1:
            best_test_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(
                classifier.state_dict(), f"checkpoints/{exp_name}/models/best_model.t7"
            )
            
            out_df = pd.DataFrame({'y_true': test_true,
                           'y_pred': test_pred})
            out_df.to_csv(f"checkpoints/{exp_name}/output/output.csv", index=False)
            
            cm = confusion_matrix(y_true=out_df["y_true"], y_pred=out_df["y_pred"])
            oa = accuracy_score(y_true=out_df["y_true"], y_pred=out_df["y_pred"])
            f1 = f1_score(y_true=out_df["y_true"], y_pred=out_df["y_pred"], average="weighted")
            recall = recall_score(y_true=out_df["y_true"], y_pred=out_df["y_pred"], average="weighted")
            precision = precision_score(y_true=out_df["y_true"], y_pred=out_df["y_pred"], average="weighted")
            labels = params["classes"]
            make_confusion_matrix(
                cm, labels, normalize=True, accuracy=oa, precision=precision, recall=recall, f1=f1
            )
            
            filename = f"checkpoints/{exp_name}/output/confusion_matrix.png"
            plt.savefig(f"checkpoints/{exp_name}/output/confusion_matrix.png")
        
        # print and save losses and f1
        io.cprint(
            f"Epoch: {epoch + 1}, Training - Augmentor Loss: {train_loss_a}, Training - Classifier Loss: {train_loss_c}, Training F1: {train_f1}, Validation Loss: {test_loss}, Validation F1: {val_f1}, Best F1: {best_test_f1} at Epoch {best_epoch}"
        )
        io.cprint(f"Training Classes: {np.unique(true_pred)}, Validation Classes: {np.unique(test_pred)}")
        
        try:
            send_telegram(f"Epoch: {epoch+1}, Val F1: {val_f1}, Best F1: {best_test_f1} at Epoch {best_epoch}")
            if best_epoch == epoch + 1:
                send_photos(open(filename, 'rb'))
        except:
            pass
        

        # Apply addaptive learning
        if params["adaptive_lr"] is True:
            if val_f1 < best_test_f1:
                triggertimes += 1
                if triggertimes > params["patience"]:
                    change = 1
            else:
                triggertimes = 0
            if change == 0:
                scheduler1_a.step(val_f1)
                scheduler1_c.step(val_f1)
                io.cprint(
                    f"Augmentor LR: {scheduler1_a.optimizer.param_groups[0]['lr']}, Trigger Times: {triggertimes}, Scheduler: Plateau"
                )
                io.cprint(
                    f"Classifier LR: {scheduler1_c.optimizer.param_groups[0]['lr']}, Trigger Times: {triggertimes}, Scheduler: Plateau"
                )
            else:
                scheduler2_a.step()
                scheduler2_c.step()
                io.cprint(
                    f"Augmentor LR: {scheduler2_a.optimizer.param_groups[0]['lr']}, Scheduler: Step"
                )
                io.cprint(
                    f"Classifier LR: {scheduler2_c.optimizer.param_groups[0]['lr']}, Scheduler: Step"
                )                          

def test(params, io, test_loader):
    device = torch.device("cuda" if params["cuda"] else "cpu")

    # Load model
    if params["model"] == "dgcnn":
        model = DGCNN(params, len(params["classes"])).to(device)
    else:
        raise Exception("Model Not Implemented")
        
    # Data Parallel
    model = nn.DataParallel(model, device_ids=list(range(0, params["n_gpus"])))

    # Load Pretrained Model
    model.load_state_dict(torch.load(params["model_path"]))
    
    # Setup for Testing
    model = model.eval()
    test_true = []
    test_pred = []

    # Testing
    for data, label in tqdm(
        test_loader, desc="Testing Total: ", leave=False, colour="green"
    ):
        exp_name = params["exp_name"]
        # Get data, labels, & batch size
        data, label = (
            data.to(device),
            label.to(device).squeeze(),
        )
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        # Run Model
        output = model(data)
        
        test_true.append(torch.argmax(label, dim=1).cpu().numpy())
                
        pred_sm = F.softmax(output, dim=1)
        test_pred.append(torch.argmax(pred_sm, dim=1).cpu().numpy())
        

    # Calculate f1
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    f1 = f1_score(test_true, test_pred, average='weighted')
    
    out_df = pd.DataFrame({'y_true': test_true,
                           'y_pred': test_pred})
    out_df.to_csv(f"checkpoints/{exp_name}/output/output.csv", index=False)
    

    io.cprint(f"F1: {f1}")