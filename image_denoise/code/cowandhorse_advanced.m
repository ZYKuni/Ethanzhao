%将彩色图像转化为灰度图像
image = imread("test.PNG");
gray_image = rgb2gray(image);

%图像预处理
gray_image = double(gray_image);
maxu = max(gray_image(:));
minu = min(gray_image(:));
u = (gray_image - minu)/(maxu - minu);%噪声图像归一化

%选做部分
image_size = size(u);
noise_std = 20/255;
missing_ratio = 0.5;
%添加高斯噪声
u_noisy = u + noise_std *randn(image_size);

missing_mask = rand(image_size) > missing_ratio;
u_observed = u_noisy .* missing_mask;

%图像降噪
lambda = 1;
iters = 100;
grad_norms = zeros(4, iters, 1);
func_values = zeros(4, iters, 1);
psnr_values = zeros(4, iters, 1);
[duo, grad_norms(1,:,:), func_values(1,:,:), psnr_values(1,:,:)] = gradient_descent_BB_ad(u, u_observed, lambda, iters, missing_mask);%lambda = 1
[duo2, grad_norms(2,:,:), func_values(2,:,:), psnr_values(2,:,:)] = gradient_descent_BB(gray_image, duo, 0.7, iters);%lambda = 0.7

%呈现降噪结果
subplot(1, 3, 1);
imshow(u);
title('gray image');

subplot(1, 3, 2);
imshow(u_observed);
title('observed image');

subplot(1, 3, 3);
imshow(duo2);
title('denoised image');

draw_psnr(iters, psnr_values);