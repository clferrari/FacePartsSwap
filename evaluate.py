#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import BiSeNet


def vis_parsing_maps(im, parsing_anno, swapPartIndex, stride, save_im=False, save_path='.res/vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    face_eval_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
        if pi in swapPartIndex:
            face_eval_color[index[0], index[1], :] = [255, 255, 255]
        else:
            face_eval_color[index[0], index[1], :] = [0, 0, 0]

    # Save result or not
    if save_im:
        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return face_eval_color


def evaluate(img, respth='./res/vis_results', cp='model_final_diss.pth', use_cpu=False, saveParsingMap = False):

    if not os.path.exists(respth):
        os.makedirs(respth)

    # List of parts
    atts = ['bkgrnd', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    # Use all face
    swapPartIndex = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    n_classes = 19

    net = BiSeNet(n_classes=n_classes)

    if use_cpu:
        net.cpu()
    else:
        net.cuda()

    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(img, 'RGB')
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)

        if not use_cpu:
            img = img.cuda()

        out = net(img)[0]

        # Convert to CelebMaskHQ parsing config.
        if saveParsingMap:
            label_list = ['bkgrnd', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
                          'mouth',
                          'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
            out2 = out.clone()
            i = 0
            for c_sean, c_det in zip(label_list, atts):
                idx = atts.index(c_sean)
                out2[:, i] = out[:, idx]
                i += 1

            Image.fromarray(out2.argmax(1).permute(1, 2, 0).cpu().squeeze(2).numpy().astype(np.uint8)).save(respth + '/parsing_map.png')

        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        face_index = vis_parsing_maps(image, parsing, swapPartIndex, stride=1, save_im=saveParsingMap)

    return face_index
