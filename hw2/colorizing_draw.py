import matplotlib.pyplot as plt
import numpy as np

num_images = 4
methods = ['NCC25', 'NCC35', 'SSD25', 'SSD35']

# small size 
times = [[0.5265, 0.5264, 0.5307, 0.5401],
         [0.9634, 0.9365, 0.9135, 0.9232],
         [0.2074, 0.2227, 0.2194, 0.2273],
         [0.4806, 0.3752, 0.3640, 0.3859]
        ]

# all image
# times = [[0.5265, 62.0031, 57.8093, 55.6588, 54.7862, 0.5264, 0.5307, 55.9410, 54.8790, 0.5401, 55.6483, 54.4945, 61.3033],
#          [0.9634, 88.8769, 97.3205, 87.5673, 89.2878, 0.9365, 0.9135, 89.3492, 88.6764, 0.9232, 90.0300, 90.5078, 84.8409],
#          [0.2074, 34.6633, 42.3424, 33.3305, 34.1393, 0.2227, 0.2194, 32.8467, 31.8005, 0.2273, 32.4954, 29.5304, 32.9082],
#          [0.4806, 62.8778, 58.6454, 61.7880, 57.6643, 0.3752, 0.3640, 63.5923, 63.7467, 0.3859, 62.2611, 66.5601, 52.4260]
#         ]




x = np.arange(num_images)
bar_width = 0.2 
offsets = np.arange(0, bar_width * len(methods), bar_width) 

plt.figure(figsize=(5, 7))

for i, method in enumerate(methods):
    plt.bar(x + offsets[i], times[i], bar_width, label=method)

plt.xlabel('Image Number')
plt.ylabel('Generation Time (seconds)')
plt.title('Comparison of Generation Time for 11 Images using 4 Methods')
# np.arange(1, num_images + 1)
plt.xticks(x + bar_width * (len(methods) / 2), labels=[1, 6, 7, 10]) 
plt.legend()
plt.grid(True)
plt.show()
