%-----------特征相关性分析--------------
%% 清空环境变量
clc;close all;clear;warning off;
%% 导入数据

   x = xlsread('光谱指数整合数据 _按耐盐性排序.xlsx',1);
    data = x(3:end,1:end)
   
    n_features = 33;     % 特征数量n_features = 27;
   
   
% -----------------------------------------------------------------------------

data_normalized = zeros(size(data,1),n_features); % 预分配内存

for i = 1:n_features
    % 计算均值和标准差
    mu = mean(data(:, i));
    sigma = std(data(:, i));
    
    % Z-score 归一化
    data_normalized(:, i) = (data(:, i) - mu) / sigma;
    
   
end
 features = data_normalized(:, 1:end);
 y = data(:, end);

    %% 相关性分析

       features_corr = corr(features, y);
      %features_corr = corr(features, y, 'Type', 'Spearman');
      %  features_corr = fscmrmr(features, y);
      % features_corr=features_corr';
    pcc_save{i, 1} = features_corr;

    %% 获取前10个最相关的特征及其相关系数（可改）
    [sorted_corr, sorted_idx] = sort(abs(features_corr), 'descend');
    top10_idx = sorted_idx(1:min(15, length(sorted_idx)));  % 防止特征不足10个***
    top10_corr = features_corr(top10_idx);

    %% 保存结果到Excel
    result_table = table((1:length(top10_idx))', top10_idx, top10_corr, ...
        'VariableNames', {'排名', '特征编号', '相关系数'});
    writetable(result_table, sprintf('Top15相关特征_数据集%d.xlsx', i));

    %% 可视化
    figure('Position', [100, 100, 800, 600], 'Color', 'w');

    % 绘制所有特征相关性柱状图
    subplot(2, 1, 1);
    bar(features_corr, 'FaceColor', [0.4, 0.6, 0.8], 'EdgeColor', 'none');
    hold on;
    % 标记前10个特征
    scatter(top10_idx, top10_corr, 100, 'r', 'filled');
    title(sprintf('Feature importance analysis (the red dots represent the top 15)', i), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Feature Number', 'FontSize', 10);
    ylabel('MRMR Score', 'FontSize', 10);
    grid on;
    set(gca, 'FontSize', 9, 'LineWidth', 1.2);
    xlim([0, n_features+1]);

    % 单独绘制Top10特征
    subplot(2, 1, 2);
    barh(top10_corr(end:-1:1), 'FaceColor', [0.8, 0.4, 0.4], 'EdgeColor', 'none');
    set(gca, 'YTick', 1:length(top10_idx), 'YTickLabel', num2str(top10_idx(end:-1:1)));
    title('Top15重要特征', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('MRMR分数', 'FontSize', 10);
    ylabel('特征编号', 'FontSize', 10);
    grid on;
    set(gca, 'FontSize', 9, 'LineWidth', 1.2, 'XAxisLocation', 'top');

    % 调整布局并保存图片
    set(gcf, 'Color', 'w');
    exportgraphics(gcf, sprintf('特征重要性分析_数据集%d.png', i), 'Resolution', 600);

    
    xlswrite('提取的重要特征数据.xlsx',x(:,[1,11,12,15,19,22,25,27,28,30]));   % 导出至Excel表格