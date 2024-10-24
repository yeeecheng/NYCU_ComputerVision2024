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
        self.ga_pyramid = [self.image]
        self.ga_spectrum = []
    
    def run(self):
        # Generate Gaussian pyramid with a specified number of levels using cv2.pyrDown
        self.ga_spectrum = [self.magnitude_spectrum(self.image)]
        current_image = self.image

        for _ in range(1, self.level):
            current_image = cv2.pyrDown(current_image)  # Use cv2 to downsample the image
            self.ga_pyramid.append(current_image)
            self.ga_spectrum.append(self.magnitude_spectrum(current_image))
        
        self.visualize()

    def magnitude_spectrum(self, img):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        spectrum = 20 * np.log(np.abs(fshift) + 1)
        return spectrum

    def visualize(self):
        # Visualize each level of the pyramid
        num_levels = len(self.ga_pyramid)
        fig, axs = plt.subplots(num_levels, 2, figsize=(8, 4 * num_levels))

        for i in range(num_levels):
            axs[i, 0].imshow(self.ga_pyramid[i], cmap='gray')
            axs[i, 0].set_title(f'Gaussian Level {i + 1}')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(self.ga_spectrum[i], cmap='gray')
            axs[i, 1].set_title('Magnitude Spectrum')
            axs[i, 1].axis('off')

        plt.tight_layout()
        # plt.savefig(self.output)
        plt.show()


def list_files_in_directory(directory):
    try:
        files = os.listdir(directory)
        files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
        return files
    except Exception as e:
        print(f"Error: {e}")
        return []

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--data', default="./data/task1and2_hybrid_pyramid", type=str, help='path of data dir')
    parse.add_argument('-o', '--output', default="./output/0_Afghan_girl_after", type=str, help='path of output data dir')
    parse.add_argument('-l', '--level', default=5, type=int, help='Number of image levels.')
    args = parse.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    directory_path = args.data 
    file_list = list_files_in_directory(directory_path)

    for file_name in file_list:
        args = parse_args()
        args.data = f"./data/task1and2_hybrid_pyramid/{file_name}"
        args.output = f"./output_cv2/{file_name.split('/')[-1].split('.')[0]}"

        gaussian_pyramid = GaussianPyramid(args)
        gaussian_pyramid.run()
