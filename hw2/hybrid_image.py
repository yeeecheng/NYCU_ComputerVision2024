import os, math, cv2
import numpy as np
import matplotlib.pyplot as plt

# fig_manager = plt.get_current_fig_manager()
# fig_manager.window.showMaximized()


def preprocess(data1, data2):
    data1 = data1[:-1,1:-1,:]
    data2 = data2[:, :-1, :]
    return data1, data2

def hybrid(data1, data2, cf1, cf2, Filter):
    h, w, c = data1.shape
    low_pass_result = convolution(data1, Filter(h, w, cf1, 'low-pass'))
    high_pass_result = convolution(data2, Filter(h, w, cf2, 'high-pass'))
    hybrid_result = low_pass_result + high_pass_result
    return hybrid_result, low_pass_result, high_pass_result

def convolution(img, H):
    h, w, c = img.shape
    result = np.zeros((h,w,c))
    
    for c_tmp in range(0, c):
        img_tmp = img[:, :, c_tmp].copy() / 255
        # step 1
        for i in range(0, h):
            for j in range(0, w):
                img_tmp[i, j] = img_tmp[i, j] * ((-1) ** (i + j))
        # step 2
        F = Fourier_transformation(img_tmp)
        # step 3
        tmp = F * H
        # step 4 & 5
        result[:, :, c_tmp] = inverse_Fourier_transformation(tmp).real
        # step 6
        for i in range(0, h):
            for j in range(0, w):
                result[i, j, c_tmp] = result[i, j, c_tmp] * ((-1) ** (i + j))
    return result

def Fourier_transformation(img):
    return np.fft.fft2(img)

def inverse_Fourier_transformation(img):
    return np.fft.ifft2(img)

def ideal_filter(h, w, cf, mode):
    x0, y0 = w // 2, h // 2
    tmp, H = (1, np.zeros((h, w))) if mode == 'low-pass' else (0, np.ones((h, w)))
    
    for x in range(x0 - cf, x0 + cf):
        y_min = max(int(y0 - math.sqrt(cf ** 2 - (x - x0) ** 2)), 0)
        y_max = min(int(y0 + math.sqrt(cf ** 2 - (x - x0) ** 2)), h)
        for y in range(y_min, y_max):
            H[y, x] = tmp
            
    return H

def Gaussian_filter(h, w, cf, mode):
    x0, y0 = w // 2, h // 2
    tmp, H = (1, np.zeros((h, w))) if mode == 'low-pass' else (-1, np.ones((h, w)))
    
    for x in range(w):
        for y in range(h):
            H[y, x] += (math.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / (2 * cf ** 2))) * tmp
            
    return H

def normalize(img):
    return ((img - img.min()) / (img.max() - img.min()))


def plot(position, title, image):
    plt.subplot(position)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

def save_fig(data1, data2, low_pass_result, high_pass_result, result, img_name, cf1, cf2, filename):
    position = [231, 234, 232, 235, 133]
    title = ['Image 1', 'Image 2',
             f'Low-pass\n(cutoff frequency = {cf1})', f'High-pass\n(cutoff frequency = {cf2})',
             'Hybrid Image']
    image = [data1[:,:,::-1], data2[:, :, ::-1], normalize(low_pass_result)[:,:,::-1],
             normalize(high_pass_result)[:,:,::-1], normalize(result)[:,:,::-1]]
    
    for i in range(0, len(position)):
        plot(position[i], title[i], image[i])
    
    plt.savefig(f'{filename}/{fig1[0: 1]}_{img_name}_cf_{cf1}_{cf2}_result.jpg')
    plt.close()


if __name__ == '__main__':
    
    # using TA's data
    """
    path = 'data/task1and2_hybrid_pyramid'
    fig_name = [['0_Afghan_girl_after.jpg', '0_Afghan_girl_before.jpg'],
                ['1_bicycle.bmp', '1_motorcycle.bmp'],
                ['2_bird.bmp', '2_plane.bmp'],
                ['3_cat.bmp', '3_dog.bmp'],
                ['4_einstein.bmp', '4_marilyn.bmp'],
                ['5_fish.bmp', '5_submarine.bmp'],
                ['6_makeup_after.jpg', '6_makeup_before.jpg']]
    """
    # using our data
    
    path = 'my_data'
    fig_name = [['tiger.png', 'panda.png'],
                ['dog.png', 'tiger.png'],
                ['cat.png', 'tiger.png']]
    

    cutoff_frequency_low = [6, 7, 8, 9, 10, 11, 12, 13]
    cutoff_frequency_high = [6, 7, 8, 9, 10, 11, 12, 13]
    
    filename = '1_result'
    if not os.path.exists(filename):
        os.makedirs(filename) 
    
    for i in range(0, len(fig_name)):
        fig1, fig2 = fig_name[i][0], fig_name[i][1]
        data1, data2 = cv2.imread(path + '/' + fig1), cv2.imread(path + '/' + fig2)
        
        if fig1 == '6_makeup_after.jpg':
            data1, data2 = preprocess(data1, data2)
        
        for cf_low in cutoff_frequency_low:
            for cf_high in cutoff_frequency_high:
                result, low_pass_result, high_pass_result = hybrid(data1, data2, cf_low, cf_high, ideal_filter)
                save_fig(data1, data2, low_pass_result, high_pass_result, result, 'ideal', cf_low, cf_high, filename)
                result, low_pass_result, high_pass_result = hybrid(data1, data2, cf_low, cf_high, Gaussian_filter)
                save_fig(data1, data2, low_pass_result, high_pass_result, result, 'Gaussian', cf_low, cf_high, filename)


    