import random
import sys
import time
import argparse
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from BiSIC_models import *
from lib.utils import AverageMeter, StereoImageDataset, MSE_Loss_Stereo, get_output_folder, save_checkpoint
import numpy as np
import wandb
import os
from tqdm import tqdm
from pytorch_msssim import ms_ssim

os.environ["WANDB_API_KEY"] = ""  # write your own wandb id
# os.environ["WANDB_MODE"] = "offline"

def compute_aux_loss(aux_list: List, backward=False):
    # a function to compute and backward aux_loss
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()

    return aux_loss_sum


def configure_optimizers(net, learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=learning_rate,
    )

    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, Lambda):
    model.train()
    device = next(model.parameters()).device
    # define three names for wandb recording
    metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
    metric_name = "mse_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter('mse', ':.4e')
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = 'mse' + "0", 'mse' + "1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')
    # these lines create some AverageMeters with Name and value

    train_dataloader = tqdm(train_dataloader)
    print('Train epoch:', epoch)
    for i, batch in enumerate(train_dataloader):
        # extract the two frames in batch
        data_in = [frame.to(device) for frame in batch]
        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()
        # aux_optimizer.zero_grad()

        out_net = model(data_in)
        out_criterion = criterion(out_net, data_in, Lambda)

        out_criterion["loss"].backward()
        # clip the gradient with 1.0, in case of gradient exposion
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if aux_optimizer is not None:
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
            # notice that the backward of aux_loss is included in compute_aux_loss
            aux_optimizer.step()
        else:
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)

        # These lines update the AverageMeters
        loss.update(out_criterion["loss"].item())
        bpp_loss.update((out_criterion["bpp_loss"]).item())
        aux_loss.update(out_aux_loss.item())
        metric_loss.update(out_criterion[metric_name].item())

        left_bpp.update(out_criterion["bpp0"].item())
        right_bpp.update(out_criterion["bpp1"].item())

        if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
            # mse0 and mse1 > 0, calculate the dB version of metrics
            left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
            right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
            left_db.update(left_metric)
            right_db.update(right_metric)
            metric_dB.update((left_metric + right_metric) / 2)

        train_dataloader.set_description('[{}/{}]'.format(i, len(train_dataloader)))
        # show the situation of losses of training
        train_dataloader.set_postfix(
            {"Loss": loss.avg, 'Bpp': bpp_loss.avg, 'mse': metric_loss.avg, 'Aux': aux_loss.avg,
            metric_dB_name: metric_dB.avg})
    # return the losses
    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg,
            "aux_loss": aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg,
            "right_bpp": right_bpp.avg,
            left_db_name: left_db.avg, right_db_name: right_db.avg, }

    return out


def test_epoch(epoch, val_dataloader, model, criterion, Lambda):
    model.eval()
    device = next(model.parameters()).device
    metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
    metric_name = "mse_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter('mse', ':.4e')
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = 'mse' + "0", 'mse' + "1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')
    loop = tqdm(val_dataloader)

    with torch.no_grad():
        for i, batch in enumerate(loop):
            d = [frame.to(device) for frame in batch]

            out_net = model(d)
            out_criterion = criterion(out_net, d, Lambda)

            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)

            loss.update(out_criterion["loss"].item())
            bpp_loss.update((out_criterion["bpp_loss"]).item())
            aux_loss.update(out_aux_loss.item())
            metric_loss.update(out_criterion[metric_name].item())

            left_bpp.update(out_criterion["bpp0"].item())
            right_bpp.update(out_criterion["bpp1"].item())

            if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
                left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
                right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
                left_db.update(left_metric)
                right_db.update(right_metric)
                metric_dB.update((left_metric + right_metric) / 2)

            loop.set_description('[{}/{}]'.format(i, len(val_dataloader)))
            loop.set_postfix({"Loss": loss.avg, 'Bpp': bpp_loss.avg, 'mse': metric_loss.avg, 'Aux': aux_loss.avg,
                                metric_dB_name: metric_dB.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg,
            "aux_loss": aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg,
            "right_bpp": right_bpp.avg,
            left_db_name: left_db.avg, right_db_name: right_db.avg, }

    return out


def train_BiSIC(epochs, device, batch_size, test_batch_size, learning_rate, dataset_name, dataset_path, output_path, load_model_path, Lambda, model_name, savecheckpoint, clip_max_norm=1.0):
    # load the datasets
    train_dataset = StereoImageDataset(ds_type='train', ds_name=dataset_name, root=dataset_path, crop_size=(256,256), resize=False)
    test_dataset = StereoImageDataset(ds_type='test', ds_name=dataset_name, root=dataset_path, crop_size=(256,256), resize=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=2, shuffle=False, pin_memory=(device == "cuda"))

    # decide the network 
    if model_name == 'BiSIC':
        net = BiSIC_master(N=192, M=192)
    elif model_name == 'BiSIC_Fast':
        net = BiSIC_Fast(N=192, M=192)
    else:
        raise ValueError("Model name not recognized")
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150, 250, 350, 550], 0.5)
    criterion = MSE_Loss_Stereo()  # notice that lambda is given in forward()
    last_epoch = 0
    best_loss = float("inf")
    
    if load_model_path:  # load from previous checkpoint
        print("Loading model: ", load_model_path)
        checkpoint = torch.load(load_model_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_b_model_path = os.path.join(os.path.split(load_model_path)[0], 'ckpt.best.pth.tar')
        best_loss = torch.load(best_b_model_path)["loss"]
        
    log_dir, experiment_id = get_output_folder(
        f'{output_path}/Results{dataset_name}/{model_name}/lamda{int(Lambda)}/', 'train')
    display_name = "{}_lmbda{}".format(model_name, int(Lambda))
    tags = "lmbda{}".format(Lambda)

    project_name = model_name + "_ECCV2024_" + dataset_name
    wandb.init(project=project_name, name=display_name, tags=[tags], )
    wandb.watch_called = False  # Re-run the model without re-starting the runtime, unnecessary after our next release
    # wandb.config.update(argses)  # config is a variable that holds and saves hyper parameters and inputs
    metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
    metric_name = "mse_loss"

    for epoch in range(last_epoch, epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, Lambda)
        lr_scheduler.step()

        wandb.log({"train": {"loss": train_loss["loss"], metric_name: train_loss[metric_name],
                            "bpp_loss": train_loss["bpp_loss"],
                            "aux_loss": train_loss["aux_loss"], metric_dB_name: train_loss[metric_dB_name],
                            "left_bpp": train_loss["left_bpp"], "right_bpp": train_loss["right_bpp"],
                            left_db_name: train_loss[left_db_name], right_db_name: train_loss[right_db_name]}, })
        # for every 10 epochs, test it on the test dataset
        if epoch % 10 == 0:
            # validation loss
            val_loss = test_epoch(epoch, test_dataloader, net, criterion, Lambda)
            wandb.log({
                "test": {"loss": val_loss["loss"], metric_name: val_loss[metric_name], "bpp_loss": val_loss["bpp_loss"],
                        "aux_loss": val_loss["aux_loss"], metric_dB_name: val_loss[metric_dB_name],
                        "left_bpp": val_loss["left_bpp"], "right_bpp": val_loss["right_bpp"],
                        left_db_name: val_loss[left_db_name], right_db_name: val_loss[right_db_name], }})

            loss = val_loss["loss"]
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        else:
            loss = best_loss
            is_best = False
        if savecheckpoint:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                is_best, log_dir
            )


def main():
    parser = argparse.ArgumentParser(description="Training script for BiSIC")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=850, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate at beginning of training')
    parser.add_argument('--dataset_name', type=str, default='instereo2k', help='Name of dataset')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save output')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load the pre-trained checkpoint if any')
    parser.add_argument('--Lambda', type=int, default=256, help='Lambda value in R-D loss')
    parser.add_argument('--model_name', type=str, default='BiSIC', help='Name of the model to train')
    parser.add_argument('--savecheckpoint', action='store_true', help='Save checkpoint during training')

    args = parser.parse_args()

    train_BiSIC(
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        learning_rate=args.learning_rate,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        load_model_path=args.load_model_path,
        model_name=args.model_name,
        Lambda=args.Lambda,
        savecheckpoint=args.savecheckpoint
    )

if __name__ == "__main__":
    main()