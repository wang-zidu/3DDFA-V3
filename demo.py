import argparse
import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image

from face_box import face_box
from model.recon import face_model
from util.preprocess import get_data_path
from util.io import visualize
def main(args):

    recon_model = face_model(args)
    facebox_detector = face_box(args).detector
    im_path = get_data_path(args.inputpath)

    for i in range(len(im_path)):
        print(i, im_path[i])
        im = Image.open(im_path[i]).convert('RGB')
        trans_params, im_tensor = facebox_detector(im)

        recon_model.input_img = im_tensor.to(args.device)
        results = recon_model.forward()

        if not os.path.exists(os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg',''))):
            os.makedirs(os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg','')))
        my_visualize = visualize(results, args)

        my_visualize.visualize_and_output(trans_params, cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), \
            os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg','')), \
            im_path[i].split('/')[-1].replace('.png','').replace('.jpg',''))
        # my_visualize.visualize_and_output(trans_params, cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), \
        #     os.path.join(args.savepath), \
        #     im_path[i].split('/')[-1].replace('.png','').replace('.jpg',''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA-V3')

    parser.add_argument('-i', '--inputpath', default='examples/', type=str,
                        help='path to the test data, should be a image folder')
    parser.add_argument('-s', '--savepath', default='examples/results', type=str,
                        help='path to the output directory, where results (obj, png files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cuda or cpu' )

    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped and resized into (224,224,3).' )
    parser.add_argument('--detector', default='retinaface', type=str,
                        help='face detector for cropping image, support for mtcnn and retinaface')

    # save
    parser.add_argument('--ldm68', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 68 landmarks')
    parser.add_argument('--ldm106', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks')
    parser.add_argument('--ldm106_2d', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks, face profile is in 2d form')
    parser.add_argument('--ldm134', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 134 landmarks' )
    parser.add_argument('--seg', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d without visible mask' )
    parser.add_argument('--seg_visible', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d with visible mask' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save obj use texture from BFM model')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save obj use texture extracted from input image')

    main(parser.parse_args())