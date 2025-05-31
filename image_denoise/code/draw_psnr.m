function draw_psnr(iters, psnr_values)
%函数用于绘制迭代过程中不同lambda值的收敛性
    
    % 正则化系数
    lambda_values = [0.1, 0.4, 0.7, 1];
    %lambda_values = [1, 1];

    % 存储每个 lambda 的结果
    psnr_values_all = cell(length(lambda_values), 1);

    for i = 1:length(lambda_values)
        psnr_values_all{i} = psnr_values(i,:,:);
    end
    % 绘制图像
    figure;
    % 绘制psnr图
    subplot(1, 1, 1);
    hold on;
    for i = 1:length(lambda_values)
        semilogy(1:iters, psnr_values_all{i}, 'DisplayName', ['\lambda = ' num2str(lambda_values(i))]);
    end
    hold off;
    xlabel('Iteration');
    ylabel('psnr');
    legend;
    title('psnr');
end