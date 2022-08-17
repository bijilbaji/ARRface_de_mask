import torch
from models.face_encoder_res50 import FaceEncoder
from models.face_decoder import Face3D
from utils.cv import img2tensor, tensor2img
from utils.simple_renderer import SimpleRenderer
from utils.align import Preprocess
import cv2
import numpy as np
import os
import pickle
import argparse
# import facerec as fr

fpath = os.path.dirname(os.path.realpath(__file__))
root_img = fpath +'/input/'
root_ldmk = fpath +'/ldmk/'
result_path = fpath +'/result/'

img_lst = os.listdir(root_img)
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
args, unknown = parser.parse_known_args()
input_image = args.image
filename_full = os.path.basename(input_image)
filename = filename_full[:filename_full.rfind(".")]
#  code exicution example
#  python /content/ARRface_de_mask/test.py --image /content/ARRface_de_mask/input/0.png
#  python test.py --image "G:\Google Drive\13iG\Projects\Current\Face Recgonisson With Mask Removal\git\face_de_mask/input/0.png"


def load_model():
    face_encoder = torch.nn.DataParallel(FaceEncoder().cuda(), [0])
    state_dict = torch.load(
        fpath + '/ckpt/it_200000.pkl', map_location='cpu')
    face_encoder.load_state_dict(state_dict)
    face_decoder = torch.nn.DataParallel(Face3D().cuda(), [0])
    tri = face_decoder.module.facemodel.tri.unsqueeze(0)

    renderer = torch.nn.DataParallel(SimpleRenderer().cuda(), [0])
    return face_encoder.eval(), face_decoder.eval(), tri, renderer


face_encoder, face_decoder, tri, renderer = load_model()

for name in img_lst:
    img_pth = os.path.join(root_img, name)
    pkl_name = name.split('.')[0] + '.pkl'
    ldmk_pth = os.path.join(root_ldmk, pkl_name)
    with open(ldmk_pth, 'rb') as f:
        ldmk = pickle.load(f)

    I = cv2.imread(input_image)[:, :, ::-1]
    J, new_ldmk = Preprocess(I, ldmk)
    J_tensor = img2tensor(J)

    with torch.no_grad():
        coeff = face_encoder(J_tensor)
        verts, tex, id_coeff, ex_coeff, tex_coeff, gamma, keypoints = face_decoder(
            coeff)
        out = renderer(verts, tri, size=256, colors=tex, gamma=gamma)
        recon = out['rgb']

    recon = tensor2img(recon)
    show = np.concatenate((J, recon), 1)
    cv2.imwrite(result_path + filename + "_result.jpg", show[..., ::-1])
    cv2.imwrite(result_path + filename + "_output.jpg", recon)
    break
