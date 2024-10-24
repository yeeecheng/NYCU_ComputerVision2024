import argparse
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import time

def split_image_to_BGR(img):
    """ Split the original image into three images each of RGB"""
    h = img.shape[0]
    h_unit = h // 3
    # B, G, R
    return [img[: h_unit], img[h_unit : 2 * h_unit], img[2 * h_unit : 3 * h_unit]]

def crop_img(org_img, divided_img, crop_ratio= 0.1):
    """ Crop the image to reduce noise or some unwanted black borders"""
    h, w = org_img.shape[:2]
    h_unit, w_unit = h // 3, w
    crop_h_unit, crop_w_unit = int(h_unit * crop_ratio), int(w_unit * crop_ratio)
    
    cropped_img = []
    for img in divided_img:
        cropped_img.append(
            img[crop_h_unit : h_unit - crop_h_unit, crop_w_unit : w_unit - crop_w_unit]
        )
    return cropped_img

def ncc(img1, img2):
    """ normalize cross correlation """
    
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    
    numerator = np.sum((img1 - mean1) * (img2 - mean2))
    denominator = np.sqrt(np.sum((img1 - mean1) ** 2) * np.sum((img2 - mean2) ** 2))
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

def ssd(img1, img2):
    """ Sum of Squared Differences (SSD) """
    
    # difference of two images 
    difference = np.array(img1, dtype=np.float64) - np.array(img2, dtype=np.float64)
    squared_difference = difference ** 2
    ssd_value = np.sum(squared_difference)
    
    return -ssd_value

def align_channels(base_channel, offset_channel, loss= "ncc", max_shift= 25):
    """ Align B, G, R channel """
    best_shift = (0, 0)
    best_score = None
    # run all shift
    for x_shift in range(-max_shift, max_shift + 1):
        for y_shift in range(-max_shift, max_shift + 1):
            # shift x and y
            shifted_channel = np.roll(offset_channel, x_shift, axis= 1)
            shifted_channel = np.roll(shifted_channel, y_shift, axis= 0)
            # calculate ncc 
            if loss == "ncc":
                loss_score = ncc(base_channel, shifted_channel)
            # calculate ssd 
            elif loss == "ssd":
                loss_score = ssd(base_channel, shifted_channel)
            # record best 
            if best_score is None or loss_score > best_score:
                best_score = loss_score
                best_shift = (x_shift, y_shift)
                # print(max_ncc, best_shift)
    return best_shift

def colorizing(args, img):
    
    # pre-processing image
    divided_img = split_image_to_BGR(img)
    cropped_B, cropped_G, cropped_R = crop_img(img, divided_img, args.crop_ratio)
    cropped_h, cropped_w = cropped_B.shape[:2]
    
    # downsample img to reduce computation cost, then find best shift in downsampled img.
    down_scale = args.down_scale
    down_B = cv2.resize(cropped_B, (cropped_h // down_scale, cropped_w // down_scale))
    down_G = cv2.resize(cropped_G, (cropped_h // down_scale, cropped_w // down_scale))
    down_R = cv2.resize(cropped_R, (cropped_h // down_scale, cropped_w // down_scale))
    green_shift = align_channels(down_B, down_G, args.loss_function, args.shift)
    red_shift = align_channels(down_B, down_R, args.loss_function, args.shift)

    print(f"green shift(x, y) {green_shift[0] * down_scale}, {green_shift[1] * down_scale}")
    print(f"red shift(x, y) {red_shift[0] * down_scale}, {red_shift[1] * down_scale}")
    # align G, R image with down scale multiply shift of green and red.
    aligned_G = np.roll(cropped_G, green_shift[0] * down_scale, axis= 1)
    aligned_G = np.roll(aligned_G, green_shift[1] * down_scale, axis= 0)
    aligned_R = np.roll(cropped_R, red_shift[0] * down_scale, axis= 1)
    aligned_R = np.roll(aligned_R, red_shift[1] * down_scale, axis= 0)

    # concate RGB channel to RGB image.
    return np.dstack([aligned_R, aligned_G, cropped_B])

def save_and_visual(img, file_name, visual= False):
    
    plt.imshow(img)
    plt.axis('off')
    if not os.path.isdir("./res"):
        os.mkdir("./res")
    plt.savefig(f"./res/great_{file_name}", bbox_inches='tight', pad_inches=0)
    if visual:
        plt.show()

def run(args):

    dir = args.data_path
    data_list = os.listdir(dir)

    for file_name in data_list:
        
        img = plt.imread(os.path.join(dir, file_name))
        splited_file_name = file_name.split(".")
        file_name = splited_file_name[0]
        print(f"file name: {file_name}")
        # because type of tif file is u64
        if splited_file_name[1] == "tif":
            img = img.astype(np.float64)
            img /= 65535.0

        start_time = time.time()
        RGB_img = colorizing(args, img)
        end_time = time.time()
        save_and_visual(RGB_img, file_name)
        print(f"using time: {end_time - start_time} s")
        print("----------------------------------------------")

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--data_path', type= str, default= "./data/task3_colorizing/", help='Data Path')
    parse.add_argument('-c', '--crop_ratio', type= float, default= 0.1, help= "ratio of crop img board.")
    parse.add_argument('-D', '--down_scale', type= int, default= 5, help= 'scale of downsampling')
    parse.add_argument('-l', '--loss_function', type= str, default= 'ncc', choices= ['ssd', 'ncc'], help= 'the method which evaluate two image whether are aligned')
    parse.add_argument('-s', '--shift', type= int, default= 35, help= 'number of roll shift between -shift and shift.')
    args = parse.parse_args() 
    return args


if __name__ == "__main__":

    args = parse_args()
    run(args)