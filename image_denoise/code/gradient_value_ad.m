function [value, grad] = gradient_value_ad(x, y, O, lambda)
% 计算损失函数值和梯度，用于带有缺失值的图像优化
% 输入：x：当前图像矩阵，y：观测的有噪声图像（含缺失值）；O：二值矩阵，观测到的位置值为1，缺失位置值为0；lambda：正则化系数
% 输出：value：损失函数值，grad：梯度矩阵

    % 核范数梯度计算
    [U, S, V] = svd(x, 'econ');
    S_threshold = max(S - lambda, 0); % 奇异值软阈值化
    x_nuclear = U * S_threshold * V';
    
    value = 0.5 * norm(O .* (y - x), 'fro')^2 + lambda * sum(diag(S)); % 损失函数值的计算

    % 梯度计算
    grad = O .* (x - y) + (x - x_nuclear); 
end
