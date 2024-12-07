% 讀取 Python 儲存的 .mat 文件
data = load('data.mat');

% 提取變數
P = data.P;
p_img2 = data.p_img2;
M = data.M;
tex_name = data.tex_name;
im_index = data.im_index;
output_dir = data.output_dir;

% 呼叫 obj_main 函數
obj_main(P, p_img2, M, "./data/Mesona1.JPG", 1, "./");