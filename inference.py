import math
import sys
import os
import time
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm
import compressai

from compressai.zoo.pretrained import load_pretrained
from BiSIC_models import BiSIC_master, BiSIC_Fast
from lib.utils import CropCityscapesArtefacts, MinimalCrop



def collect_images(data_name:str, rootpath: str):
    # extract the folder path and files
    if data_name == 'cityscapes':
        left_image_list, right_image_list = [], []
        path = Path(rootpath)
        for left_image_path in path.glob(f'leftImg8bit/test/*/*.png'):
            left_image_list.append(str(left_image_path))
            right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))

    elif data_name == 'instereo2k':
        path = Path(rootpath)
        path = path / "test"   
        folders = [f for f in path.iterdir() if f.is_dir()]
        left_image_list = [f / 'left.png' for f in folders]
        right_image_list = [f / 'right.png' for f in folders] #[1, 3, 860, 1080], [1, 3, 896, 1152]

    elif data_name == 'wildtrack':
        C1_image_list, C4_image_list = [], []
        path = Path(rootpath)
        for image_path in path.glob(f'images/C1/*.png'):
            if int(image_path.stem) > 2000:
                C1_image_list.append(str(image_path))
                C4_image_list.append(str(image_path).replace("C1", 'C4'))
        left_image_list, right_image_list = C1_image_list, C4_image_list

    return [left_image_list, right_image_list]


def read_image(crop_transform, filepath: str) -> torch.Tensor:
    # take the image file and crop it
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    if crop_transform is not None:
        img = crop_transform(img)
    return transforms.ToTensor()(img)


def compute_metrics_for_frame(
    org_frame: Tensor,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,):
    
    org_frame = (org_frame * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_frame - rec_frame).pow(2).mean()
    psnr_float = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)
    ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=max_val)
    return psnr_float, ms_ssim_float


def compute_metrics_for_frame_withMSE(
    org_frame: Tensor,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,):

    org_frame = (org_frame * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_frame - rec_frame).pow(2).mean()
    psnr_float = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)
    ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=max_val)
    return mse_rgb, psnr_float, ms_ssim_float

def compute_bpp(likelihoods, num_pixels):
    bpp = sum(
        (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
        for likelihood in likelihoods.values()
    )
    return bpp


def run_inference(
    filepaths,
    netCompressor: nn.Module, 
    outputdir: Path, data_name, crop_needed,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = ""):
    
    # this function is just a translator, to run with entropy estimation or not
    left_filepath, right_filepath = filepaths[0], filepaths[1]
    with torch.no_grad():
        if entropy_estimation:
            metrics = eval_model_entropy_estimation(netCompressor, left_filepath, right_filepath, data_name, crop_needed)
        else:
            raise NotImplementedError
            # metrics = eval_model_compress_myConv3D(netCompressor, left_filepath, right_filepath, data_name, crop_needed)
    return metrics


@torch.no_grad()
def eval_model_entropy_estimation(netCompressor:nn.Module, left_filepaths: Path, right_filepaths: Path, data_name, crop_needed) -> Dict[str, Any]:
    device = next(netCompressor.parameters()).device
    num_frames = len(left_filepaths)
    print(f'Evaluating on a dataset with {num_frames} images...') 
    max_val = 2**8 - 1
    results = defaultdict(list)
    if crop_needed:
        crop_transform = CropCityscapesArtefacts() if data_name == "cityscapes" else MinimalCrop(min_div=64)
        # crop for test dataset depending on the dataset type
    else:
        crop_transform = None
    
    results = defaultdict(list)
    
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            x_left = read_image(crop_transform, left_filepaths[i]).unsqueeze(0).to(device)
            num_pixels = x_left.size(2) * x_left.size(3)

            x_right = read_image(crop_transform, right_filepaths[i]).unsqueeze(0).to(device)
            left_height, left_width = x_left.shape[2:]
            right_height, right_width = x_right.shape[2:]
            # read the images and get the size
            # rectify the code here if you want to save the visualization

            out = netCompressor([x_left, x_right])
            x_left_rec, x_right_rec = out["x_hat"][0], out["x_hat"][1]
            left_likelihoods, right_likelihoods = out["likelihoods"][0], out["likelihoods"][1]
            
            x_left_rec = x_left_rec.clamp(0, 1)
            x_right_rec = x_right_rec.clamp(0, 1)

            metrics = {}
            metrics["left-mse"], metrics["left-psnr-float"], metrics["left-ms-ssim-float"] = compute_metrics_for_frame_withMSE(
            x_left, x_left_rec, device, max_val)
            metrics["right-mse"], metrics["right-psnr-float"], metrics["right-ms-ssim-float"] = compute_metrics_for_frame_withMSE(
            x_right, x_right_rec, device, max_val)
            metrics["left-ms-ssim-dB"] = 10 * torch.log10(1 / (1 - metrics["left-ms-ssim-float"]))
            metrics["right-ms-ssim-dB"] = 10 * torch.log10(1 / (1 - metrics["right-ms-ssim-float"]))
            
            metrics["mse-average"] = (metrics["left-mse"] + metrics["right-mse"])/2
            metrics["psnr-float"] = (metrics["left-psnr-float"]+metrics["right-psnr-float"])/2
            metrics["ms-ssim-float"] = (metrics["left-ms-ssim-float"]+metrics["right-ms-ssim-float"])/2
            metrics["ms-ssim-dB"] = (metrics["left-ms-ssim-dB"] + metrics["right-ms-ssim-dB"])/2

            metrics["left_bpp"] = compute_bpp(left_likelihoods, num_pixels)
            metrics["right_bpp"] = compute_bpp(right_likelihoods, num_pixels)
            metrics["bpp"] = (metrics["left_bpp"] + metrics["right_bpp"])/2
            
            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results



def eval_BiSIC(device, dataset_dir, data_name, load_model_path, choose_entropy_estimation, output_path, crop_needed=True, network_decision='Main'):
    description = ("entropy-estimation" if choose_entropy_estimation else compressai.available_entropy_coders()[0])
    filepaths = collect_images(data_name, dataset_dir)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)
    
    if device == "cpu":
        cpu_num = 2 # change it to the target number of CPUs
        os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num)
    # load the network here
    if network_decision == 'Main':
        netCompressor = BiSIC_master(N=192, M=192)
        description_name = 'BiSIC'
    elif network_decision == 'Fast':
        netCompressor = BiSIC_Fast(N=192, M=192)
        description_name = 'BiSIC_Fast'
    else:
        print("Error: no network found.", file=sys.stderr)
        raise SystemExit(1)
    netCompressor = netCompressor.to(device)
    if load_model_path:
        print("Loading model:", load_model_path)
        checkpoint = torch.load(load_model_path, map_location=device)
        netCompressor.load_state_dict(checkpoint["state_dict"])
        netCompressor.update(force=True)
        netCompressor.eval()
    
    outputdir = output_path
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    results = defaultdict(list)
    trained_net = f"{description_name}-{description}"
    metrics = run_inference(filepaths, netCompressor, outputdir, entropy_estimation=choose_entropy_estimation,
                            trained_net=trained_net, description=description, data_name=data_name, crop_needed=True)
    for k, v in metrics.items():
        results[k].append(v)

    output = {
        "name": f"{description_name}",
        "description": f"Inference ({description})",
        "results": results,
    }

    with (Path(f"{outputdir}/{description_name}-{description}.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inference script for BiSIC")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the pre-trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output metrics')
    parser.add_argument('--data_name', type=str, default='instereo2k', help='Name of the dataset')
    parser.add_argument('--choose_entropy_estimation', action='store_true', help='Use evaluated entropy estimation as bit calculation')
    parser.add_argument('--network', type=str, default='Main', help='Decide the network to use')

    args = parser.parse_args()

    eval_BiSIC(
        device=args.device,
        dataset_dir=args.dataset_dir,
        data_name=args.data_name,
        load_model_path=args.model_dir,
        choose_entropy_estimation=args.choose_entropy_estimation,
        output_path=args.output_dir,
        network_decision=args.network
    )


if __name__ == "__main__":
    main()