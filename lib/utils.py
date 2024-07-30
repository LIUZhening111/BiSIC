import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import torchvision.transforms.functional as tf
from typing import Dict
import numpy as np
import math
from PIL import Image
import random
import torch
from skimage import morphology
from pytorch_msssim import ms_ssim
from .tools_SSIM import MS_SSIM


class CropCityscapesArtefacts:
    """Crop Cityscapes images to remove artefacts"""
    def __init__(self):
        # define the area of target crop
        self.top = 64
        self.left = 128
        self.right = 128
        self.bottom = 256

    def __call__(self, image):
        """Crops a PIL image.
        Args:
            image (PIL.Image): Cityscapes image (or disparity map)
        Returns:
            PIL.Image: Cropped PIL Image
        """
        w, h = image.size
        assert w == 2048 and h == 1024, f'Expected (2048, 1024) image but got ({w}, {h}). Maybe the ordering of transforms is wrong?'
        #w, h = 1792, 704
        # return image.crop((self.left, self.top, w-self.right, h-self.bottom))
        return transforms.functional.crop(image, self.top, self.left, h-self.bottom, w-self.right)

class MinimalCrop:
    """
    Performs the minimal crop such that height and width are both divisible by min_div. For test dataset only.
    """
    # this is used to corresond to the limitation of test, it can only work on divided sizes
    def __init__(self, min_div=16):
        self.min_div = min_div
        
    def __call__(self, image):
        w, h = image.size
        # calculate the extra size that can not be divided
        h_new = h - (h % self.min_div)
        w_new = w - (w % self.min_div)
        
        if h_new == 0 and w_new == 0:
            # successfully get divided int
            return image
        else:    
            h_diff = h-h_new
            w_diff = w-w_new
            # these lines average the extra part to be two parts, so it is centered
            top = int(h_diff/2)
            bottom = h_diff-top
            left = int(w_diff/2)
            right = w_diff-left

            return image.crop((left, top, w-right, h-bottom))
        
class StereoImageDataset(Dataset):
    """Dataset class for image compression datasets."""
    def __init__(self, ds_type='train', ds_name='instereo2k', root='/Prac_MIC_SASIC/Instereo2K', crop_size=None, resize=False, **kwargs):
        """
        Args:
            name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
            path (str): if given the dataset is loaded from path instead of by name.
            transforms (Transform): transforms to apply to image
            debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
        """
        super().__init__()
        
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        # if it is for train, crop a patch, else only to tensor
        if ds_type=="train":
            if crop_size == None:
                print('Notice: The full image is fed into training.')
                self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
            else: 
                self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.left_image_list, self.right_image_list = self.get_files()


        if ds_name == 'cityscapes':
            self.crop = CropCityscapesArtefacts()
        else:
            if ds_type == "test":
                self.crop = MinimalCrop(min_div=64)
            else:
                self.crop = None
        #self.index_count = 0

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.left_image_list)} files.')

    def __len__(self):
        return len(self.left_image_list)

    def __getitem__(self, index):
        #self.index_count += 1
        image_list = [Image.open(self.left_image_list[index]).convert('RGB'), Image.open(self.right_image_list[index]).convert('RGB')]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), 2)
        if random.random() < 0.5:
            frames = frames[::-1]
        return frames

    def get_files(self):
        if self.ds_name == 'cityscapes':
            left_image_list, right_image_list, disparity_list = [], [], []
            for left_image_path in self.path.glob(f'leftImg8bit/{self.ds_type}/*/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))
                disparity_list.append(str(left_image_path).replace("leftImg8bit", 'disparity'))

        elif self.ds_name == 'instereo2k':
            path = self.path / self.ds_type
            # if self.ds_type == "test":
            #     folders = [f for f in path.iterdir() if f.is_dir()]
            # else:
            #     folders = [f for f in path.glob('*/*') if f.is_dir()]
            folders = [f for f in path.iterdir() if f.is_dir()]
            # because the dataset is modified and organized, no glob is needed
            left_image_list = [f / 'left.png' for f in folders]
            right_image_list = [f / 'right.png' for f in folders]

        elif self.ds_name == 'kitti':
            left_image_list, right_image_list = [], []
            ds_type = self.ds_type + "ing"
            for left_image_path in self.path.glob(f'stereo2012/{ds_type}/colored_0/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("colored_0", 'colored_1'))

            for left_image_path in self.path.glob(f'stereo2015/{ds_type}/image_2/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("image_2", 'image_3'))

        elif self.ds_name == 'wildtrack':
            C1_image_list, C4_image_list = [], []
            for image_path in self.path.glob(f'images/C1/*.png'):
                if self.ds_type == "train" and int(image_path.stem) <= 2000:
                    C1_image_list.append(str(image_path))
                    C4_image_list.append(str(image_path).replace("C1", 'C4'))
                elif self.ds_type == "test" and int(image_path.stem) > 2000:
                    C1_image_list.append(str(image_path))
                    C4_image_list.append(str(image_path).replace("C1", 'C4'))
            left_image_list, right_image_list = C1_image_list, C4_image_list
        else:
            raise NotImplementedError

        return left_image_list, right_image_list


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    
class MSE_Loss_Stereo(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        # self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # loss calculation
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2
        out["mse0"] = self.mse(output['x_hat'][0], target1)
        out["mse1"] = self.mse(output['x_hat'][1], target2)
        
        if isinstance(lmbda, list):
            out['mse_loss'] = (lmbda[0] * out["mse0"] + lmbda[1] * out["mse1"])/2 
        else:
            out['mse_loss'] = lmbda * (out["mse0"] + out["mse1"])/2        #end to end
        out['loss'] = out['mse_loss'] + out['bpp_loss']

        return out
    
class MS_SSIM_Loss_Stereo(nn.Module):
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val, device_id=int(device[5:])).to(device)
        
    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # loss calculation
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2
        
        out["ms_ssim0"] = 1 - self.ms_ssim(output['x_hat'][0], target1)
        out["ms_ssim1"] = 1 - self.ms_ssim(output['x_hat'][1], target2)
        
        out['ms_ssim_loss'] =  (out["ms_ssim0"] + out["ms_ssim1"]) / 2      
        out['loss'] = lmbda * out['ms_ssim_loss'] + out['bpp_loss']

        return out
    
    
def get_output_folder(parent_dir, env_name, output_current_folder=False):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    if not output_current_folder: 
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir, experiment_id


def save_checkpoint(state, is_best=False, log_dir=None, filename="ckpt.pth.tar"):
    save_file_dir = os.path.join(log_dir, filename)
    print("save model in:", save_file_dir)
    torch.save(state, save_file_dir)
    if is_best:
        torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out
    
    
class RB(nn.Module):
	def __init__(self, channels):
		super(RB, self).__init__()

		self.body = nn.Sequential(
			nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
		)
	def forward(self, x):
		return self.body(x) + x



def morphologic_process(mask):
	device = mask.device
	b,_,_,_ = mask.shape

	mask = ~mask
	mask_np = mask.cpu().numpy().astype(bool)
	mask_np = morphology.remove_small_objects(mask_np, 20, 2)
	mask_np = morphology.remove_small_holes(mask_np, 10, 2)

	for idx in range(b):
		buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
		buffer = morphology.binary_closing(buffer, morphology.disk(3))
		mask_np[idx,0,:,:] = buffer[3:-3,3:-3]

	mask_np = 1-mask_np
	mask_np = mask_np.astype(float)

	return torch.from_numpy(mask_np).float().to(device)

class SAM(nn.Module):
	def __init__(self, channels):
		super(SAM, self).__init__()

		self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
		self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
		self.rb = RB(channels)
		self.softmax = nn.Softmax(-1)
		self.bottleneck = nn.Conv2d(channels * 2+1, channels, 1, 1, 0, bias=True)

	def forward(self, x_left, x_right):
		b, c, h, w = x_left.shape
		buffer_left = self.rb(x_left)
		buffer_right = self.rb(x_right)

		### M_{right_to_left}
		Q = self.b1(buffer_left).permute(0, 2, 3, 1)  # B * H * W * C
		S = self.b2(buffer_right).permute(0, 2, 1, 3)  # B * H * C * W
		score = torch.bmm(Q.contiguous().view(-1, w, c),
						  S.contiguous().view(-1, c, w))  # (B*H) * W * W
		M_right_to_left = self.softmax(score)

		score_T = score.permute(0,2,1)
		M_left_to_right = self.softmax(score_T)

		# valid mask
		V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
		V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
		V_left_to_right = morphologic_process(V_left_to_right)
		V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
		V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
		V_right_to_left = morphologic_process(V_right_to_left)

		buffer_R = x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
		buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

		buffer_L = x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
		buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

		out_L = self.bottleneck(torch.cat((buffer_l, x_left, V_left_to_right), 1))
		out_R = self.bottleneck(torch.cat((buffer_r, x_right, V_right_to_left), 1))

		return out_L, out_R
