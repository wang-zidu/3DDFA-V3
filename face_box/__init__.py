import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
import os
import cv2
from mtcnn import MTCNN
from util.preprocess import load_lm3d, align_img
from .facelandmark.large_model_infer import LargeModelInfer

def no_crop(im):
    if np.array(im).shape==(224,224,3):
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return None, im
    else:
        print('the original face image should be well cropped and resized to (224,224,3).')
        exit()

class retinaface:
    def __init__(self):
        # retinaface uses cuda
        self.landmark_model = LargeModelInfer("assets/large_base_net.pth", device='cuda')
        self.lm3d_std = load_lm3d()

    def detector(self, im):
        img = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        H = img.shape[0]
        _, results_all = self.landmark_model.infer(img)
        if len(results_all)>0:
            results = results_all[0] # only use the first one
            landmarks=[]
            for idx in [74, 83, 54, 84, 90]:
                landmarks.append([results[idx][0], results[idx][1]])
            landmarks = np.array(landmarks).astype(np.float32)
            landmarks[:, -1] = H - 1 - landmarks[:, -1]

            trans_params, im, lm, _ = align_img(im, landmarks, self.lm3d_std)

            im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            return trans_params, im

        else:
            print('no face detected! run original image')
            if np.array(im).shape==(224,224,3):
                im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                return None, im
            else:
                print('exit. no face detected! run original image. the original face image should be well cropped and resized to (224,224,3).')
                exit()

class mtcnnface:
    def __init__(self):
        self.landmark_model = MTCNN()
        self.lm3d_std = load_lm3d()

    def detector(self, im):
        img = np.asarray(im)
        H = img.shape[0]
        facial_landmarks = self.landmark_model.detect_faces(img)

        if len(facial_landmarks)>0:
            # Find the landmark with the highest confidence
            highest_confidence_landmark = max(facial_landmarks, key=lambda x: x['confidence'])
            if highest_confidence_landmark['confidence'] > 0.6:
                landmarks = []
                for key, value in highest_confidence_landmark['keypoints'].items():
                    landmarks.append([value[0], value[1]])
                landmarks=np.array(landmarks).astype(np.float32)
                # print(landmarks)

                trans_params, im, lm, _ = align_img(im, landmarks, self.lm3d_std)

                im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                return trans_params, im

            else:
                print('no face detected! run original image')
                if np.array(im).shape==(224,224,3):
                    im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                    return None, im
                else:
                    print('exit. no face detected! run original image. the original face image should be well cropped and resized to (224,224,3).')
                    exit()

        else:
            print('no face detected! run original image')
            if np.array(im).shape==(224,224,3):
                im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                return None, im
            else:
                print('exit. no face detected! run original image. the original face image should be well cropped and resized to (224,224,3).')
                exit()

class face_box:
    def __init__(self, args):
        if args.iscrop:
            if args.detector == 'mtcnn':
                m = mtcnnface()
                self.detector = m.detector
                print('use mtcnn for face box')
            elif args.detector == 'retinaface':
                r = retinaface()
                self.detector = r.detector
                print('use retinaface for face box')
            else:
                print('please check the detector')
                exit()
        else:
            print('run original image in (224,224,3) size')
            self.detector = no_crop
