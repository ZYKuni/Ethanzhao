function [denoised_image, grad_norms, func_values, psnr_values] = gradient_descent_BB(gray_image, y, lambda, iters)
%梯度下降采用BB步长实现
%输入：噪声图像y，正则化系数lambda，最大迭代次数iters，原始图像gray_image(用于计算psnr)
    %初始化
    x = y;
    psnr_values = zeros(iters, 1);
    grad_norms = zeros(iters, 1);
    func_values = zeros(iters, 1);
    step = 2;
    for iter = 1:iters
        %计算函数值及梯度
        [value, grad] = gradient_value(x, y, lambda);

        %记录梯度的范数和函数值
        grad_norms(iter) = norm(grad, 'fro');
        func_values(iter) = value;
        %if (grad_norms(iters) / grad_norms(1) < 1e-6)
        %    break

        %BB步长
        if iter > 1
            s = x - x_prev;
            y_diff = grad - grad_prev;
            step = (s(:)' * s(:)) / (s(:)' * y_diff(:) + eps);
        end

        %更新x
        x_prev = x;
        grad_prev = grad;
        x = x - step * grad;

        %计算当前psnr
        psnr_values(iter) = PSNR(gray_image, x*255, 255);
        

        %可视化结果
        %if mod (iter, 10) == 0
            %imshow(x,[]);
            %title(['迭代次数: ', num2str(iter), ' - PSNR: ', num2str(psnr_values(iter))]);
            %drawnow;
        %end
    end
    denoised_image = x;
end