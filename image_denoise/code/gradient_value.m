function [value, grad] = gradient_value(x, y, lambda)
%函数用于计算函数值和梯度，用于后面的梯度下降算法
%输入：x，y，lambda（正则系数）
%输出：函数值value，梯度grad
    %差分矩阵计算
    D1x = x(:, [2:end, end]) - x;
    D2x = x([2:end, end], :) - x;

    %函数值计算
    value = 0.5 * norm(x - y, 'fro')^2 + lambda * (norm(D1x, 'fro')^2 + norm(D2x, 'fro')^2);

    %梯度计算
    L_x = x([2:end, end], :) + x([1, 1:end-1], :) + x(:, [2:end, end]) + x(:, [1, 1:end-1]) - 4 * x;
    grad = x - y - 2 * lambda * L_x;
end