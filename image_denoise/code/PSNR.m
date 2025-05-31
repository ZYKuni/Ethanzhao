function psnrValue = PSNR(image1, image2, maxPixelValue)
    % 将图像转换为双精度
    image1 = double(image1);
    image2 = double(image2);
    
    % 计算均方误差（Mean Squared Error，MSE）
    mse = mean((image1(:) - image2(:)).^2);
    
    % 计算PSNR
    psnrValue = 10 * log10((maxPixelValue^2) / mse);
end