import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt

def gaussian_blur(img):
    
    # create a 5 * 5 Gaussian kernel
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], dtype=np.float32)
    # Normalize the kernel (make the sum of elements equal to 1)
    kernel /= np.sum(kernel)
    
    img_h, img_w = img.shape[:2]
    kernel_h, kernel_w = kernel.shape
    # Padding size 
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    # Padding image
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    # Create output image base on original img size
    blur_img = np.zeros_like(img)
    
    for y in range(img_h):
        for x in range(img_w):
            # B, G, R channel
            for c in range(3):
                blur_img[y, x, c] = np.sum(padded_img[y : y + kernel_h, x : x + kernel_w, c] * kernel)
                
    return blur_img

def downsampling(img):
    # Downsample the image by taking every second pixel in both dimensions
    return img[::2, ::2]

def gaussian_pyramid(img, levels):
    
    pyramid = [img]
    for i in range(levels - 1):
        # blur_img = gaussian_blur(pyramid[i])
        # pyramid.append(downsampling(blur_img))
        pyramid.append(cv2.pyrDown(pyramid[i]))
    return pyramid

def show_different_scaled_pyramid(file_name, pyramid):
    # output different scale images
    for i, layer in enumerate(pyramid):
        # cv2.imwrite(f'./pyramid_result/{file_name}_Level_{i}_{layer.shape[1]}x{layer.shape[0]}.png', layer)
        cv2.imwrite(f'./{file_name}_Level_{i}_{layer.shape[1]}x{layer.shape[0]}.png', layer)

def show_adjust_size_pyramid(file_name, pyramid):
    
    for i, layer in enumerate(pyramid):
        plt.subplot(1, levels, i + 1)
        plt.imshow(cv2.cvtColor(layer, cv2.COLOR_BGR2RGB))
        plt.title(f'Level {i}')
        plt.axis('off')
    # plt.savefig(f"./pyramid_result/{file_name}.png")
    plt.savefig(f"./{file_name}.png")
        
dir = "./data/task1and2_hybrid_pyramid/"
data_list = os.listdir(dir)
for file_name in data_list:
    img = cv2.imread(os.path.join(dir, file_name))
    file_name = file_name.split(".")[0]
    levels = 5
    pyramid = gaussian_pyramid(img, levels)
    show_different_scaled_pyramid(file_name, pyramid)
    show_adjust_size_pyramid(file_name, pyramid)