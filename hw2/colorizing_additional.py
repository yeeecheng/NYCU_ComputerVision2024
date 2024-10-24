import argparse
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
from colorizing import *

def sobel(img):

    gray_image = (img * 255).astype(np.uint8)

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5) 
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  

    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255)) 

    return sobel_magnitude

def save_sobel(img, file_name):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if not os.path.isdir("./res"):
        os.mkdir("./res")
    plt.savefig(f"./res/sobel_{file_name}", bbox_inches='tight', pad_inches=0)

def colorizing_with_sobel(args, img, file_name):
    
    # pre-processing image
    divided_img = split_image_to_BGR(img)
    cropped_B, cropped_G, cropped_R = crop_img(img, divided_img, args.crop_ratio)
    cropped_h, cropped_w = cropped_B.shape[:2]
    
    # downsample img to reduce computation cost, then find best shift in downsampled img.
    down_scale = args.down_scale
    down_B = cv2.resize(cropped_B, (cropped_h // down_scale, cropped_w // down_scale))
    down_G = cv2.resize(cropped_G, (cropped_h // down_scale, cropped_w // down_scale))
    down_R = cv2.resize(cropped_R, (cropped_h // down_scale, cropped_w // down_scale))

    # Do edge filter
    sobel_B = sobel(down_B)
    sobel_G = sobel(down_G)
    sobel_R = sobel(down_R)
    save_sobel(sobel_R, 'R_' + file_name)
    save_sobel(sobel_G, 'G_' + file_name)
    save_sobel(sobel_B, 'B_' + file_name)
    green_shift = align_channels(sobel_B, sobel_G, args.loss_function, args.shift)
    red_shift = align_channels(sobel_B, sobel_R, args.loss_function, args.shift)

    print(f"green shift(x, y) {green_shift[0] * down_scale}, {green_shift[1] * down_scale}")
    print(f"red shift(x, y) {red_shift[0] * down_scale}, {red_shift[1] * down_scale}")
    # align G, R image with down scale multiply shift of green and red.
    aligned_G = np.roll(cropped_G, green_shift[0] * down_scale, axis= 1)
    aligned_G = np.roll(aligned_G, green_shift[1] * down_scale, axis= 0)
    aligned_R = np.roll(cropped_R, red_shift[0] * down_scale, axis= 1)
    aligned_R = np.roll(aligned_R, red_shift[1] * down_scale, axis= 0)

    # concate RGB channel to RGB image.
    return np.dstack([aligned_R, aligned_G, cropped_B])

def run(args):
    # data_path = "./data/task3_colorizing/emir.tif"
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
        RGB_img = colorizing_with_sobel(args, img, file_name)
        end_time = time.time()
        save_and_visual(RGB_img, file_name)
        print(f"using time: {end_time - start_time} s")
        print("----------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    run(args)