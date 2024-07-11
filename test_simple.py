from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
import cv2
import heapq
from PIL import ImageFile, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--output_path', type=str,
                        help='path to a folder of output images', required=True)

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        )

    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        )

    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="smtadapter",
                        choices=[
                            "litemono",
                            "smt",
                            "smtadapter",
                            "monovit",
                            "hrdepth",
                            "monodepth"])

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    # assert args.load_weights_folder is not None, \
    #     "You must specify the --load_weights_folder parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model == "smt":
        args.load_weights_folder = "mypretrain/smt_b_640/"
    elif args.model == "smtadapter":
        args.load_weights_folder = "mypretrain/smt_b_adapter_640/"
    elif args.model == "monovit":
        args.load_weights_folder = "mypretrain/MonoViT_M_640x192/"
    elif args.model == "hrdepth":
        args.load_weights_folder = "mypretrain/HR_Depth_CS_K_MS_640x192/"
    elif args.model == "litemono":
        args.load_weights_folder = "mypretrain/lite-mono-8m_640x192/"
    else:
        args.load_weights_folder = "mypretrain/mono_640x192/"

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    # encoder = networks.LiteMono(model=args.model,
    #                                 height=feed_height,
    #                                 width=feed_width)
    if args.model == "smt":
        encoder = networks.smt_b()
    elif args.model == "smtadapter":
        encoder = networks.SMTAdapter()
    elif args.model == "monovit":
        encoder = networks.mpvit_small()
    elif args.model == "litemono":
        encoder = networks.LiteMono()
    else:
        encoder = networks.ResnetEncoder(18, False)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    if args.model == "smtadapter":
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
    elif args.model == "monovit":
        depth_decoder = networks.MonoVitDepthDecoder()
    elif args.model == "hrdepth":
        depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, scales=range(4))
    elif args.model == "smt":
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
    elif args.model == "litemono":
        depth_decoder = networks.MonoDepthDecoder(encoder.num_ch_enc, scales=range(3))
    else:
        depth_decoder = networks.M2DepthDecoder(encoder.num_ch_enc, scales=range(4))

    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path) and not args.test:
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.output_path)
    elif os.path.isfile(args.image_path) and args.test:
        gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        # reading images from .txt file
        paths = []
        with open(args.image_path) as f:
            filenames = f.readlines()
            for i in range(len(filenames)):
                filename = filenames[i]
                line = filename.split()
                folder = line[0]
                if len(line) == 3:
                    frame_index = int(line[1])
                    side = line[2]

                f_str = "{:010d}{}".format(frame_index, '.jpg')
                image_path = os.path.join(
                    'kitti_data',
                    folder,
                    "image_0{}/data".format(side_map[side]),
                    f_str)
                paths.append(image_path)

    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.output_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            # input_image = input_image.crop((0, 750, 1704, 1300)).resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            if args.model == "hrdepth":
                disp = outputs[("disparity", "Scale0")]
            else:
                disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            disp_resized = disp

            # attn_mask = get_local.cache['Attention.atten']
            # print(len(attn_mask), "  attn layer", attn_mask[13].shape, " attn shape")
            # print(input_image.size())
            # image = pil.open(image_path)
            # visualize_grid_to_grid(attn_mask[13][0,0,1:,1:], 100, image)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # output_name = os.path.splitext(image_path)[0].split('/')[-1]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if args.model == "monodepth":
                sx = "m2"
            elif args.model == "hrdepth":
                sx = "hr"
            elif args.model == "monovit":
                sx = "mnv"
            elif args.model == "smtadapter":
                sx = "smta"
            elif args.model == "litemono":
                sx = "lite"
            else:
                sx = "smt"

            # name_dest_npy = os.path.join(output_directory, "{}_{}_disp.npy".format(output_name, sx))
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im).convert("L")

            name_dest_im = os.path.join(output_directory, "depth{}.png".format(output_name[4:]))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            # print("   - {}".format(name_dest_npy))
    print('-> Done!')


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape
    with_cls_token = False

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = pil.fromarray(mask).resize((image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
