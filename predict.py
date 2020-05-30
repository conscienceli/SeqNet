############Test
import argparse
import os
import tensorflow as tf
from keras.backend import tensorflow_backend

from utils import define_model, crop_prediction
from keras.layers import ReLU
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2

from PIL import Image


def predict(ACTIVATION='ReLU', dropout=0.1, batch_size=32, repeat=4, minimum_kernel=32, 
            epochs=200, iteration=3, crop_size=128, stride_size=3, 
            input_path='', output_path='', DATASET='ALL'):
    exts = ['png', 'jpg', 'tif', 'bmp', 'gif']

    if not input_path.endswith('/'):
        input_path += '/'
    paths = [input_path + i for i in sorted(os.listdir(input_path)) if i.split('.')[-1] in exts]

    gt_list_out = {}
    pred_list_out = {}

    os.makedirs(f"{output_path}/out_seg/", exist_ok=True)
    os.makedirs(f"{output_path}/out_art/", exist_ok=True)
    os.makedirs(f"{output_path}/out_vei/", exist_ok=True)
    os.makedirs(f"{output_path}/out_final/", exist_ok=True)

    activation = globals()[ACTIVATION]
    model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration)
    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
    print("Model : %s" % model_name)
    load_path = f"trained_model/{DATASET}/{model_name}.hdf5"
    model.load_weights(load_path, by_name=False)

    for i in tqdm(range(len(paths))):
        filename = '.'.join(paths[i].split('/')[-1].split('.')[:-1])
        img = Image.open(paths[i])
        image_size = img.size
        img = np.array(img) / 255.
        img = resize(img, [576, 576])

        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img, crop_size, stride_size)
        preds = model.predict(patches_pred)

        #for segmentation
        pred = preds[iteration]
        pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
        pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
        pred_imgs = pred_imgs[:, 0:576, 0:576, :]
        probResult = pred_imgs[0, :, :, 0]
        pred_ = probResult
        pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
        pred_seg = pred_
        pred_ = resize(pred_, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_seg/{filename}.png", pred_)
    
        #for artery
        pred = preds[2*iteration + 1]
        pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
        pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
        pred_imgs = pred_imgs[:, 0:576, 0:576, :]
        probResult = pred_imgs[0, :, :, 0]
        pred_ = probResult
        pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
        pred_art = pred_
        pred_ = resize(pred_, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_art/{filename}.png", pred_)

        #for vein
        pred = preds[3*iteration + 2]
        pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
        pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
        pred_imgs = pred_imgs[:, 0:576, 0:576, :]
        probResult = pred_imgs[0, :, :, 0]
        pred_ = probResult
        pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
        pred_vei = pred_
        pred_ = resize(pred_, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_vei/{filename}.png", pred_)

        #for final
        pred_final = np.zeros((*list(pred_seg.shape), 3), dtype=pred_seg.dtype)
        art_temp = pred_final[pred_art >= pred_vei]
        art_temp[:,2] = pred_seg[pred_art >= pred_vei]
        pred_final[pred_art >= pred_vei] = art_temp
        vei_temp = pred_final[pred_art < pred_vei]
        vei_temp[:,0] = pred_seg[pred_art < pred_vei]
        pred_final[pred_art < pred_vei] = vei_temp
        pred_ = pred_final
        pred_ = resize(pred_, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_final/{filename}.png", pred_)




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)


    # define the program description
    des_text = 'Please use -i to specify the input dir and -o to specify the output dir.'

    # initiate the parser
    parser = argparse.ArgumentParser(description=des_text)
    parser.add_argument('--input', '-i', help="(Required) Path of input dir")
    parser.add_argument('--output', '-o', help="(Optional) Path of output dir")
    args = parser.parse_args()

    if not args.input:
        print('Please specify the input dir with -i')
        exit(1)

    input_path = args.input

    if not args.output:
        output_path = './output/'
    else:
        output_path = args.output
        if output_path.endswith('/'):
            output_path = output_path[:-1]


    #stride_size = 3 will be better, but slower
    predict(batch_size=24, epochs=200, iteration=3, stride_size=3, crop_size=128, 
        input_path=input_path, output_path=output_path)