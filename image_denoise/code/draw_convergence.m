function draw_convergence(iters, grad_norms, func_values)
%函数用于绘制迭代过程中不同lambda值的收敛性
    
    % 正则化系数
    %lambda_values = [0.1, 0.4, 0.7, 1];
    lambda_values = [1, 1];
    %stepvalues = [0.1, 0.3];

    % 存储每个 lambda 的结果
    grad_norms_all = cell(length(lambda_values), 1);
    func_values_all = cell(length(lambda_values), 1);

    for i = 1:length(lambda_values)
        
        grad_norms_all{i} = log10(grad_norms(i,:,:));
        func_values_all{i} = log10(func_values(i,:,:));
    end
    % 绘制图像
    figure;
    % 绘制梯度范数的对数图
    subplot(1, 2, 1);
    hold on;
    for i = 1:length(lambda_values)
        semilogy(1:iters, grad_norms_all{i}, 'DisplayName', ['\lambda = ' num2str(lambda_values(i))]);
        %semilogy(1:iters, grad_norms_all{i}, 'DisplayName', ['step = ' num2str(stepvalues(i))]);
    end
    hold off;
    xlabel('Iteration');
    ylabel('Norm of the gradient (log)');
    legend;
    title('Gradient Norm vs Iteration');

    % 绘制目标函数值的对数图
    subplot(1, 2, 2);
    hold on;
    for i = 1:length(lambda_values)
        semilogy(1:iters, func_values_all{i}, 'DisplayName', ['\lambda = ' num2str(lambda_values(i))]);
        %semilogy(1:iters, func_values_all{i}, 'DisplayName', ['step = ' num2str(stepvalues(i))]);
    end
    hold off;
    xlabel('Iteration');
    ylabel('f(x) (log)');
    legend;
    title('Objective Function Value vs Iteration'); 
end