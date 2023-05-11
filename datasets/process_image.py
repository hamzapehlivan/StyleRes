import numpy as np
import torch
from PIL import Image
try: 
    import dlib
except:
    print("dlib library is not installed, image alignment cannot be used.")
import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage

class ImageProcessor():
    def __init__(self, predictor_path=None) -> None:
        self.predictor = None
        if predictor_path:
            self.predictor =  dlib.shape_predictor(predictor_path)
    
    @staticmethod
    def preprocess_image(image, is_batch=True):
        image = image.resize( (256, 256))
        image = np.asarray(image).transpose(2, 0, 1).astype(np.float32) # C,H,W -> H,W,C
        image = torch.FloatTensor(image.copy())
        image = (image - 127.5) / 127.5     # Normalize
        if not is_batch:
            image = image.unsqueeze(0)
        return image

    """
        Input: A numpy image with shape NxCxHxW.
        Output: Output image with NxHxWxC with values between 0-255
    """
    @staticmethod
    def postprocess_image(image, min_val=-1.0, max_val=1.0, is_batch=True):
        image = image.astype(np.float64)
        image = (image - min_val) * 255 / (max_val - min_val)
        image = np.clip(image + 0.5, 0, 255).astype(np.uint8)
        image = image.transpose(0, 2, 3, 1)
        if not is_batch:
           image = Image.fromarray(image[0])
        return image

    """
    brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
    author: lzhbrian (https://lzhbrian.me)
    date: 2020.1.5
    note: code is heavily borrowed from
        https://github.com/NVlabs/ffhq-dataset
        http://dlib.net/face_landmark_detection.py.html
    requirements:
        apt install cmake
        conda install Pillow numpy scipy
        pip install dlib
        # download face landmark model from:
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    """

    def get_landmark(self, image):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        detector = dlib.get_frontal_face_detector()

        # img = dlib.load_rgb_image(filepath)
        img = np.asarray(image)
        dets = detector(img, 1)

        for k, d in enumerate(dets):
            shape = self.predictor(img, d)

        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm

    def align_face(self, img):
        """
        :param image: PIL image
        :return: PIL Image
        """
        if self.predictor is None:
            return img

        lm = self.get_landmark(img)

        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # read image
        # img = PIL.Image.open(filepath)

        output_size = 1024
        transform_size = 1024
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        return img