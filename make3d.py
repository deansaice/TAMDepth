from layers import disp_to_depth
import networks
import cv2
import os
import torch
import scipy.misc
from scipy import io
import numpy as np

# load_weights_folder = '/home/lsk/MonocularDepthEsitimation/smtmono/smt/smtadapter-w9/weights_9/'
# load_weights_folder = '/home/lsk/MonocularDepthEsitimation/smtmono/smt/smtatd-a100/models2/weights_7/'
load_weights_folder = '/home/lsk/MonocularDepthEsitimation/smtmono/mypretrain/lite-mono-8m_640x192'
main_path = '/home/lsk/Dataset/make3d'
encoder_path = os.path.join(load_weights_folder, "encoder.pth")
decoder_path = os.path.join(load_weights_folder, "depth.pth")
encoder_dict = torch.load(encoder_path)
decoder_dict = torch.load(decoder_path)
encoder = networks.LiteMono()
depth_decoder = networks.MonoDepthDecoder(encoder.num_ch_enc, scales=range(3))
# encoder = networks.SMTAdapter()
# depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

model_dict = encoder.state_dict()
encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
depth_model_dict = depth_decoder.state_dict()
depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

encoder.cuda()
encoder.eval()
depth_decoder.cuda()
depth_decoder.eval()


def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log


with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
    test_filenames = f.read().splitlines()
    # print(test_filenames)
test_filenames = map(lambda x: x[0:], test_filenames)

depths_gt = []
images = []
ratio = 2
h_ratio = 1 / (1.33333 * ratio)
color_new_height = int(1704 / 2)
depth_new_height = 21
for filename in test_filenames:
    mat = io.loadmat(os.path.join(main_path, "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
    depths_gt.append(mat["Position3DGrid"][:, :, 3])

    image = cv2.imread(os.path.join(main_path, "Test134", "img-{}.jpg".format(filename)))
    image = image[int((2272 - color_new_height) / 2):int((2272 + color_new_height) / 2), :, :]
    images.append(image[:, :, ::-1])
depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
depths_gt_cropped = map(lambda x: x[int((55 - 21) / 2):int((55 + 21) / 2), :], depths_gt)

depths_gt_cropped = list(depths_gt_cropped)
errors = []
with torch.no_grad():
    for i in range(len(images)):
        input_color = images[i]
        input_color = cv2.resize(input_color / 255.0, (640, 192), interpolation=cv2.INTER_NEAREST)

        input_color = torch.tensor(input_color, dtype=torch.float).cuda().permute(2, 0, 1)[None, :, :, :]

        output = depth_decoder(encoder(input_color))
        pred_disp, _ = disp_to_depth(output[("disp", 0)], 1e-3, 70)
        pred_disp = pred_disp.squeeze().cpu().numpy()
        depth_gt = depths_gt_cropped[i]
        depth_pred = 1 / pred_disp
        depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
        mask = np.logical_and(depth_gt > 0, depth_gt < 70)
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= np.median(depth_gt) / np.median(depth_pred)
        depth_pred[depth_pred > 70] = 70
        errors.append(compute_errors(depth_gt, depth_pred))
    mean_errors = np.mean(errors, 0)

print(("{:>8} | " * 4).format("abs_rel", "sq_rel", "rmse", "rmse_log"))
print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))