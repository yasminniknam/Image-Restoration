# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import argparse
import time

import torch
import torchvision as tv
import torch.nn.functional as F
from detection_util.util import *
from detection_models import networks
from PIL import Image, ImageFile, ImageFont
import json

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import math

def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format: Optional[str] = None,
) -> None:
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = tv.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.save(fp, format=format)

    return im



ImageFile.LOAD_TRUNCATED_IMAGES = True


def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    if full_size == "resize_256":
        return img.resize((config.image_size, config.image_size), method)

    if full_size == "scale_256":

        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def blend_mask(img, mask):

    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")


def main(config, input_images, image_names):
    # print("initializing the dataloader")

    model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )

    ## load model
    checkpoint_path = "./checkpoints/detection/FT_Epoch_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    # print("model weights loaded")

    model.to(config.GPU)
    model.eval()

    ## dataloader and transformation
    # print("directory of testing image: " + config.test_path)
    # imagelist = os.listdir(config.test_path)
    # imagelist.sort()
    total_iter = 0

    P_matrix = {}
    save_url = os.path.join(config.output_dir)
    # mkdir_if_not(save_url)

    input_dir = os.path.join(save_url, "input")
    output_dir = os.path.join(save_url, "mask")
    # blend_output_dir=os.path.join(save_url, 'blend_output')
    # mkdir_if_not(input_dir)
    # mkdir_if_not(output_dir)
    # mkdir_if_not(blend_output_dir)

    idx = 0
    input_dirs = []
    mask_dirs = []

    # for image_name in imagelist:
    # for scratch_image in input_images:
    for i in range(len(input_images)):
        scratch_image = input_images[i]
        image_name = image_names[i]
        idx += 1

        # print("processing ", image_name)

        results = []
        # scratch_file = os.path.join(config.test_path, image_name)
        # if not os.path.isfile(scratch_file):
            # print("Skipping non-file %s" % image_name)
            # continue
        # scratch_image = Image.open(scratch_file).convert("RGB")

        w, h = scratch_image.size

        transformed_image_PIL = data_transforms(scratch_image, config.input_size)

        scratch_image = transformed_image_PIL.convert("L")
        scratch_image = tv.transforms.ToTensor()(scratch_image)

        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)

        scratch_image = torch.unsqueeze(scratch_image, 0)
        scratch_image = scratch_image.to(config.GPU)

        P = torch.sigmoid(model(scratch_image))

        P = P.data.cpu()
        
        
        mask_dirs.append(save_image(
            (P >= 0.4).float(),
            os.path.join(output_dir, image_name[:-4] + ".png",),
            # os.path.join(output_dir, str(idx) + ".png",),
            nrow=1,
            padding=0,
            normalize=True,
        ))
        # transformed_image_PIL.save(os.path.join(input_dir, image_name[:-4] + ".png"))
        input_dirs.append(transformed_image_PIL)
        # transformed_image_PIL.save(os.path.join(input_dir, str(idx) + ".png"))

        # single_mask=np.array((P>=0.4).float())[0,0,:,:]
        # RGB_mask=np.stack([single_mask,single_mask,single_mask],axis=2)
        # blend_output=blend_mask(transformed_image_PIL,RGB_mask)
        # blend_output.save(os.path.join(blend_output_dir,image_name[:-4]+'.png'))

    return input_dirs, mask_dirs


def detection(input_opts, input_images, image_names):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_name', type=str, default="FT_Epoch_latest.pt", help='Checkpoint Name')

    parser.add_argument("--GPU", type=int, default=0)
    parser.add_argument("--test_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--input_size", type=str, default="scale_256", help="resize_256|full_size|scale_256")
    config = parser.parse_args(input_opts)

    return main(config, input_images, image_names)
