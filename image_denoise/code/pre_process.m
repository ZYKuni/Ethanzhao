%将彩色图像转化为灰度图像
image = imread("test.PNG");
gray_image = rgb2gray(image);

subplot(1, 3, 1);
imshow(image);
title('image');

subplot(1, 3, 2);
imshow(gray_image);
title('gray image');

imwrite(gray_image, 'gray_image.png');

%图像预处理
gray_image = double(gray_image);
noise_image = gray_image + 20*randn(size(gray_image));
maxu = max(noise_image(:));
minu = min(noise_image(:));
u = (noise_image - minu)/(maxu - minu);%噪声图像归一化
subplot(1, 3, 3)
imshow(u);
title('noise image');