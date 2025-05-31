%本脚本为大作业的主体脚本,其他.m文件一部分是函数，另一部分是部分代码的截取，用于局部分析与调试
%将彩色图像转化为灰度图像
image = imread("test.PNG");
gray_image = rgb2gray(image);

subplot(2, 3, 1);
imshow(gray_image);
title('gray image');

imwrite(gray_image, 'gray_image.png');

%图像预处理
gray_image = double(gray_image);
noise_image = gray_image + 20*randn(size(gray_image));
maxu = max(noise_image(:));
minu = min(noise_image(:));
u = (noise_image - minu)/(maxu - minu);%噪声图像归一化
subplot(2, 3, 2)
imshow(u);
title('noise image');

%图像降噪
lambda = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
iters = 150; %constant步长使用
%iters = 100; %BB步长使用
step = 0.1;
grad_norms = zeros(4, iters, 1);
func_values = zeros(4, iters, 1);
psnr_values = zeros(4, iters, 1);



%[du1, grad_norms(1,:,:), func_values(1,:,:), psnr_values(1,:,:)] = gradient_descent_BB(gray_image, u, lambda(1), iters);%lambda = 0.1
[du1, grad_norms(1,:,:), func_values(1,:,:), psnr_values(1,:,:)] = gradient_descent_constant(gray_image, u, lambda(1), iters, step);%lambda = 0.1
subplot(2, 3, 3);
imshow(du1, []);
title('λ = 0.1');

%[du4, grad_norms(2,:,:), func_values(2,:,:), psnr_values(2,:,:)] = gradient_descent_BB(gray_image, u, lambda(4), iters);%lambda = 0.4
[du4, grad_norms(2,:,:), func_values(2,:,:), psnr_values(2,:,:)] = gradient_descent_constant(gray_image, u, lambda(4), iters, step);%lambda = 0.4
subplot(2, 3, 4);
imshow(du4, []);
title('λ = 0.4');

%[du7, grad_norms(3,:,:), func_values(3,:,:), psnr_values(3,:,:)] = gradient_descent_BB(gray_image, u, lambda(7), iters);%lambda = 0.7
[du7, grad_norms(3,:,:), func_values(3,:,:), psnr_values(3,:,:)] = gradient_descent_constant(gray_image, u, lambda(7), iters, step);%lambda = 0.7
subplot(2, 3, 5);
imshow(du7, []);
title('λ = 0.7');

%[du10, grad_norms(4,:,:), func_values(4,:,:), psnr_values(4,:,:)] = gradient_descent_BB(gray_image, u, lambda(10), iters);%lambda = 1.0
[du10, grad_norms(4,:,:), func_values(4,:,:), psnr_values(4,:,:)] = gradient_descent_constant(gray_image, u, lambda(10), iters, step);%lambda = 1.0
subplot(2, 3, 6);
imshow(du10, []);
title('λ = 1.0');
%imwrite(du10, [], 'denoised_image.png');

%draw_convergence(iters, grad_norms, func_values);
%draw_psnr(iters, psnr_values);

%以下代码微调函数draw_convergence内部参数后可用于观察不同constant步长的收敛性
%[dub10, grad_norms(1,:,:), func_values(1,:,:)] = gradient_descent_constant(gray_image, u, lambda(10), iters, 0.1);%lambda = 1.0
%[duc10, grad_norms(2,:,:), func_values(2,:,:)] = gradient_descent_constant(gray_image, u, lambda(10), iters, 0.3);%lambda = 1.0
%draw_convergence(iters, grad_norms, func_values);

%以下代码微调函数draw_convergence内部参数后可用于比较constant步长与BB步长的收敛性
[dub10, grad_norms(1,:,:), func_values(1,:,:)] = gradient_descent_constant(gray_image, u, lambda(10), iters, 0.1);%lambda = 1.0
[duc10, grad_norms(2,:,:), func_values(2,:,:)] = gradient_descent_BB(gray_image, u, lambda(10), iters);%lambda = 1.0
draw_convergence(iters, grad_norms, func_values);

