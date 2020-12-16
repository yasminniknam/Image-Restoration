# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict

import data_FE
from options_FE.test_options import TestOptions
from models_FE.pix2pix_model import Pix2PixModel
from util_FE.visualizer import Visualizer
import torchvision.utils as vutils

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import torch
import math
import numpy as np
from PIL import Image, ImageFont

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
    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.save(fp, format=format)

    return im



def test_face(input_opts, input_images, image_list):
    opt = TestOptions().parse(_input_opts=input_opts)

    dataloader = data_FE.create_dataloader(opt, input_images, image_list)

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)


    single_save_url = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir, "each_img")


    if not os.path.exists(single_save_url):
        os.makedirs(single_save_url)

    fine_images = []
    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break

        generated = model(data_i, mode="inference")

        img_path = data_i["path"]

        for b in range(generated.shape[0]):
            img_name = os.path.split(img_path[b])[-1]
            save_img_url = os.path.join(single_save_url, img_name)

            fine_images.append(save_image((generated[b] + 1) / 2, save_img_url))

    return fine_images

