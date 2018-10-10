# -*- coding:utf-8 -*-
from __future__ import print_function
import argparse
import torch
from PIL import Image
from os import listdir
from os.path import join, split, splitext
import os
import numpy as np
import tifffile
from torch.autograd import Variable

# ===========================================================
# Argument settings
# ===========================================================

COLOR_MAP = [[0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 255, 0]]

def turn_label2rgb(arr):
    # print arr.max()
    r_arr = np.zeros(shape=(arr.shape[0] * arr.shape[1]))
    g_arr = np.zeros(shape=(arr.shape[0] * arr.shape[1]))
    b_arr = np.zeros(shape=(arr.shape[0] * arr.shape[1]))

    for idx in range(len(COLOR_MAP)):
        r_arr = np.where(arr.flatten() == idx , COLOR_MAP[idx][0], r_arr)
        g_arr = np.where(arr.flatten() == idx , COLOR_MAP[idx][1], g_arr)
        b_arr = np.where(arr.flatten() == idx , COLOR_MAP[idx][2], b_arr)
    # print r_arr.shape
    r_arr = np.reshape(r_arr, newshape=(arr.shape[0], arr.shape[1], 1))
    g_arr = np.reshape(g_arr, newshape=(arr.shape[0], arr.shape[1], 1))
    b_arr = np.reshape(b_arr, newshape=(arr.shape[0], arr.shape[1], 1))
    rgb_arr = np.concatenate([ b_arr, g_arr, r_arr], axis=2)
    rgb_arr = rgb_arr.astype(np.uint8)
    return rgb_arr


def preprocessing(data1):
    [r, w, b] = data1.shape
    w_size = 29
    data1_pad = np.pad(data1, ((14, 14), (14, 14), (0, 0)), 'symmetric')
    PatchImage = np.zeros([w_size, w_size, b, r*w])
    mark =0
    for i in range(r):
        for j in range(w):
            PatchImage[:, :, :, mark] = data1_pad[i: i + w_size, j: j + w_size, :]
            mark = mark + 1

    return PatchImage

def PatchTest(model, GPU_IN_USE, image_crop, crop_shape):

    data1 = preprocessing(image_crop)



    # ===========================================================
    # model import & setting
    # ===========================================================

    if GPU_IN_USE:
        model.cuda()
        # model = torch.load(args.model)

    num = data1.shape[3]

    out1 = np.zeros((num, 9))
    data1 = np.transpose(data1,axes=[3,1,2,0])
    data1 = np.transpose(data1,axes=[0,3,2,1])
    data1 = np.transpose(data1,axes=[0,2,1,3])
    print (data1.shape)
    num = 1
    temp1 = data1[0,:,:,:]
    print(temp1.shape)
    if GPU_IN_USE:
        # temp1 = Variable(ToTensor()(temp1)).view(1,-1,29,29)
        temp1 = Variable(torch.tensor(data1)).cuda()
    else :
        temp1 = Variable(torch.tensor(data1))
    print(temp1.shape)
    out = model(temp1)
    out = out.cpu()
    print(out.shape)
    print(out.data.numpy().shape)
    out1 = out.data.numpy()

 
    maxValue = np.max(out1, 1)
    maxIndex = np.zeros([out1.shape[0]]).astype(np.int32)
    for i in range(out1.shape[0]):
        temp = np.where(out1[i] == maxValue[i])
        temp = temp[0]
        temp = temp[0]
        maxIndex[i] = temp

    cnnClass = maxIndex.reshape(crop_shape)

    return cnnClass

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def load_nor_img(filepath):
    img_4 = tifffile.imread(filepath)
    img = img_4[:,:,0:3] 
    # img = misc.imread(filepath)
    max_img = img.max()
    min_img = img.min()
    nor_img = (img - min_img) / (max_img - min_img)
    return nor_img


def thumbnail_size_keep_ratio(rows, cols, max_size = 1300):
    ratio = max(float(rows)/float(max_size), float(cols)/float(max_size))
    thumb_rows = int(rows/ratio)
    thumb_cols = int(cols/ratio)
    print(thumb_rows, thumb_cols)
    return thumb_rows, thumb_cols

def isfolder( filepath ):

    save_file_path_list = filepath.split('/')

    if filepath[0] == '/' :
        save_file_path = '/'  + save_file_path_list[0]
    else:
        save_file_path = save_file_path_list[0]
    
    for idx in range(1,len(save_file_path_list)-1):
        save_file_path = os.path.join(save_file_path,save_file_path_list[idx])
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
    print(save_file_path)

    return

def convert_images(file_name, save_file_name, model_path = './model/SpatialNets_model_path.pth', is_gpu = False, process_lines = 5 ):
    """[summary]

    Arguments:
        file_name {[str]} -- [input file path ]
        save_file_name {[str]} -- [save file path: save result as rgb tif file]

    Keyword Arguments:
        model_path {str} -- [load path file path] (default: {'./model/SpatialNets_model_path.pth'})
        is_gpu {bool} -- [use gpu ] (default: {False})
        process_lines {int} -- [ processed rows num per cycle ] (default: {5})

    Returns:
        [result]{numpy array} -- [classed result: between 0 ~ 4]
    """
    if is_gpu :
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        GPU_IN_USE = torch.cuda.is_available()
    else:
        GPU_IN_USE = False
    model = torch.load(model_path, map_location=lambda storage, loc: storage)

    x = file_name
    image = load_nor_img(x)
    cropsize_r = int(process_lines)
    row = image.shape[0]
    col = image.shape[1]

    split_num = int (float(row)/float(cropsize_r) + 1.0)
    patchCNN = np.zeros([row, col]).astype(np.int32)
    for i in range(split_num):
        print('image name: ',x ,'\n  strides -  ', i)
        if i+1 == split_num:
            image_crop = image[i* cropsize_r: row, :, :]
            patchCNN[i* cropsize_r: row, :] = PatchTest(model, GPU_IN_USE, image_crop, [image_crop.shape[0], col])
        else:
            image_crop = image[i* cropsize_r: (i+1)*cropsize_r, :, :]
            patchCNN[i* cropsize_r: (i+1)*cropsize_r, :] = PatchTest(model, GPU_IN_USE, image_crop, [image_crop.shape[0], col])

    rgb_result = turn_label2rgb(patchCNN)

    isfolder(save_file_name)

    tifffile.imsave(save_file_name,rgb_result)

    return patchCNN


if __name__=='__main__':
    # row = 6908
    # cropsize_r = 60
    # split_num = int (float(row)/float(cropsize_r) + 1.0)
    # print(split_num)
    convert_images(file_name = '/storage/geocloud/test/data/原始影像数据库/GF2/L1A/PMS/4m多光谱/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.tiff',model_path='./DATA/train/SpatialNets_model_path.pth', save_file_name = '/storage/tmp/result.tif')
    # thumbnail_size_keep_ratio(6908,7300)


