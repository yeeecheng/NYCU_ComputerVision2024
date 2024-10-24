import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class GaussianPyramid:

    def __init__(self, args):
        
        self.image = cv2.imread(args.data, cv2.IMREAD_GRAYSCALE)  # Load grayscale image
        self.output = args.output
        self.level = args.level
        self.window = args.window
        self.sigma = args.sigma
        self.ga_pyramid = [self.image]
        self.ga_spectrum = []
    


    def run(self):

        self.ga_spectrum = [self.magnitude_spectrum(self.image)]
        current_image = self.image

        for _ in range(1, self.level):
            current_image = self.reduce(current_image)
            self.ga_pyramid.append(current_image)
            self.ga_spectrum.append(self.magnitude_spectrum(current_image))
        
        self.visualize()

    def gaussian_kernel(self):
        
        ax = np.linspace(-(self.window - 1) / 2., (self.window - 1) / 2., self.window)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(self.sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)


    def reduce(self, gray_img):
        
        offset = self.window // 2
        gwindow = self.gaussian_kernel() 
        height, width = gray_img.shape
        convolved_img = np.zeros((height, width))

        for i in range(offset, height - offset):
            for j in range(offset, width - offset):
                patch = gray_img[i - offset:i + offset + 1, j - offset:j + offset + 1]
                convolved_img[i, j] = np.sum(patch * gwindow)

        return convolved_img[offset:height - offset:2, offset:width - offset:2]



    def magnitude_spectrum(self, img):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        spectrum = 20 * np.log(np.abs(fshift) + 1)  
        return spectrum


    def visualize(self):

        # Visualize each level of the pyramid
        fig, axs = plt.subplots(self.level, 2, figsize=(8, self.level * 2))

        for i in range(self.level):
            axs[i, 0].imshow(self.ga_pyramid[i], cmap='gray')
            axs[i, 0].set_title('Gaussian' if i == 0 else '')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(self.ga_spectrum[i], cmap='gray')
            axs[i, 1].set_title('Spectrum' if i == 0 else '')
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.show()
        # plt.savefig(f"{self.output}_gaussian.jpg")

class LaplacianPyramid(GaussianPyramid):

    def __init__(self, args):
        super().__init__(args)
        self.la_pyramid = []
        self.la_spectrum = []


    def run(self):


        gaussian_pyramid = [self.image]
        current_image = self.image

        for _ in range(1, self.level):
            next_image = self.reduce(current_image)
            gaussian_pyramid.append(next_image)
            current_image = next_image

        laplacian_pyramid = []
        for i in range(self.level - 1):
            next_image_upsampled = self.expand(gaussian_pyramid[i + 1], gaussian_pyramid[i].shape)
            laplacian = gaussian_pyramid[i] - next_image_upsampled
            laplacian_pyramid.append(laplacian)

        laplacian_pyramid.append(gaussian_pyramid[-1])  
        self.la_pyramid = laplacian_pyramid
        self.visualize()


    def expand(self, image, target_shape):
        expanded_shape = (image.shape[0] * 2, image.shape[1] * 2)
        expanded_image = np.zeros(expanded_shape, dtype=image.dtype)
        
        expanded_image[::2, ::2] = image
        
        padded_image = np.pad(expanded_image, 
                            ((0, target_shape[0] - expanded_shape[0]), 
                            (0, target_shape[1] - expanded_shape[1])), 
                            mode='constant', constant_values=0)
        
        kernel = self.gaussian_kernel()
        expanded_blurred = self.apply_filter(padded_image, kernel) * 4
        return expanded_blurred

    
    def apply_filter(self, img, kernel):
        height, width = img.shape
        filtered_img = np.zeros((height, width))
        offset = self.window // 2

        for i in range(offset, height - offset):
            for j in range(offset, width - offset):
                patch = img[i - offset:i + offset + 1, j - offset:j + offset + 1]
                filtered_img[i, j] = np.sum(patch * kernel)

        return filtered_img

    def visualize(self):
        fig, axs = plt.subplots(len(self.la_pyramid), 2, figsize=(6, 12))

        for i, laplacian in enumerate(self.la_pyramid):
            laplacian_vis = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())

            axs[i, 0].imshow(laplacian_vis, cmap='gray')
            axs[i, 0].set_title('Laplacian' if i == 0 else '')
            axs[i, 0].axis('off')

            fft = np.fft.fft2(laplacian)
            fft_shifted = np.fft.fftshift(fft)
            spectrum = np.log(np.abs(fft_shifted) + 1e-8)  

            axs[i, 1].imshow(spectrum, cmap='gray')
            axs[i, 1].set_title('Spectrum' if i == 0 else '')
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.show()
        # plt.savefig(f"{self.output}_laplacian.jpg")


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--data', default= "./data/task1and2_hybrid_pyramid", type= str, help= 'path of data dir')
    parse.add_argument('-o', '--output', default= "./output/0_Afghan_girl_after", type= str, help= 'path of output data dir')
    parse.add_argument('-l', '--level', default= 5, type= int, help= 'Number of image level.')
    parse.add_argument('-w', '--window', default= 5, type= int, help= 'kernel length')
    parse.add_argument('-s', '--sigma', default= 1.0, type= float, help= 'kernel sigma')
    args = parse.parse_args()
    return args


def list_files_in_directory(directory):
    try:
        files = os.listdir(directory)
        files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
        return files
    except Exception as e:
        print(f"Error: {e}")
        return []
    
if __name__ == "__main__":


    args = parse_args()
    directory_path = args.data
    file_list = list_files_in_directory(directory_path)

    for file_name in file_list:
        args.data = f"./data/task1and2_hybrid_pyramid/{file_name}"
        args.output = f"./output/{file_name.split('/')[-1].split('.')[0]}"
        gaussian_pyramid = GaussianPyramid(args)
        gaussian_pyramid.run()
        pyramid = LaplacianPyramid(args)
        pyramid.run()

