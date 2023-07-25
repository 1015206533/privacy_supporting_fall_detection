# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms
import threading
import time
import cv2

import networks
from layers import disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on one Single Image.')
    parser.add_argument('--image_path', type=str,
                        default='./asserts/sample.png',
                        help='path to a test image')
    parser.add_argument('--output_path', type=str,
                        default='./asserts/sample.png',
                        help='path to a output image')
    parser.add_argument('--model_name', type=str,
                        default='weights_5f',
                        help='name of a pretrained model to use',
                        choices=[
                            "weights_3f",
                            "weights_5f",])
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

def prepare_model_for_test(args, device):
    model_path = args.model_name
    print("-> Loading model from ", model_path)
    model_path = os.path.join("ckpts", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    decoder_path = os.path.join(model_path, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc,
        scales=range(1),
    )

    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder.load_state_dict(decoder_dict)

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    return encoder, decoder, encoder_dict['height'], encoder_dict['width']

def task(input_file_dir, output_file_dir, encoder, decoder, thisH, thisW, device):
    file_list = os.listdir(input_file_dir)
    with torch.no_grad():
        # Load image and preprocess
        for file in file_list:
            #print(file)
            input_file = os.path.join(input_file_dir, file)
            output_file = os.path.join(output_file_dir, file)
            input_image = pil.open(input_file).convert('RGB')
            # extension = image_path.split('.')[-1]
            original_width, original_height = input_image.size
            input_image = input_image.resize((thisH, thisW), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            outputs = decoder(encoder(input_image))

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            scaled_disp, _ = disp_to_depth(disp, 0.1, 10)
            arr = scaled_disp.cpu().numpy()
            arr = arr.reshape(arr.shape[2:])
            arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))*65535
            arr = arr.astype(np.uint16)
            cv2.imwrite(output_file, arr)
    print('over! ', input_file_dir)
    return 

def inference(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder, decoder, thisH, thisW = prepare_model_for_test(args, device)
    image_path = args.image_path
    outputs_path = args.output_path
    print("-> Inferencing on image ", image_path)
    print("-> Inferencing output image ", outputs_path)

    file_dir_list = os.listdir(image_path)
    #print(file_dir_list)
    #threads = []
    for file_dir in file_dir_list:
        input_file_dir = os.path.join(image_path, file_dir)
        output_file_dir = os.path.join(outputs_path, file_dir)
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
        task(input_file_dir, output_file_dir, encoder, decoder, thisH, thisW, device)
        #thd = threading.Thread(target=task, args=(input_file_dir, output_file_dir, encoder, decoder, thisH, thisW, device))
        #thd.start()
        #threads.append(thd)
        #time.sleep(10)
    #for it in threads:
    #    it.join()
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    inference(args)
