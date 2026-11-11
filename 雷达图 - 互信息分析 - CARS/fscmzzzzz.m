%-----------特征相关性分析--------------
%% 清空环境变量
clc;close all;clear;warning off;

%% 导入数据
x = xlsread('光谱指数整合数据 _按耐盐性排序.xlsx',1);
data = x(3:end,1:end);

n_features = 33;     % 特征数量

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
pcc_save = features_corr;

%% 获取前10个最相关的特征及其相关系数（可改）
[sorted_corr, sorted_idx] = sort(abs(features_corr), 'descend');
top10_idx = sorted_idx(1:min(15, length(sorted_idx)));  % 防止特征不足10个***
top10_corr = features_corr(top10_idx);

%% 保存结果到Excel
result_table = table((1:length(top10_idx))', top10_idx, top10_corr, ...
    'VariableNames', {'排名', '特征编号', '相关系数'});
writetable(result_table, 'Top15相关特征.xlsx');

%% 可视化
figure('Position', [100, 100, 1400, 700], 'Color', 'w');

% 创建一个细胞数组来存储特征名称
feature_names = cell(1, 33);
feature_names{1} = '绿峰位置 Mg';
feature_names{2} = '绿峰反射率Rg';
feature_names{3} = '红谷位置 Mo';
feature_names{4} = '红谷反射率 Ro';
feature_names{5} = '近红外波段的反射率位置';
feature_names{6} = '近红外波段的反射率RNIR';
feature_names{7} = '蓝边位置λb';
feature_names{8} = '蓝边幅值Mb';
feature_names{9} = '蓝边面积Ab';
feature_names{10} = '黄边位置λy';
feature_names{11} = '黄边幅值Mb';
feature_names{12} = '黄边面积Ab';
feature_names{13} = '红边位置λr';
feature_names{14} = '红边幅值Mr';
feature_names{15} = '红边面积Ar';
feature_names{16} = '最大吸收深度Dh';
feature_names{17} = '吸收宽度W';
feature_names{18} = '吸收位置P';
feature_names{19} = '总面积S';
feature_names{20} = '左面积Sl';
feature_names{21} = '右面积Sr';
feature_names{22} = '对称度V';
feature_names{23} = '面积归一化最大吸收深度W';
feature_names{24} = '宽深比BDR';
feature_names{25} = '最大吸收深度Dh';
feature_names{26} = '吸收宽度W';
feature_names{27} = '吸收位置P';
feature_names{28} = '总面积S';
feature_names{29} = '左面积Sl';
feature_names{30} = '右面积Sr';
feature_names{31} = '对称度V';
feature_names{32} = '面积归一化最大吸收深度W';
feature_names{33} = '宽深比BDR';

% 绘制所有特征相关性柱状图
subplot(2, 1, 1);
bar(features_corr, 'FaceColor', [0.4, 0.6, 0.8], 'EdgeColor', 'none');
hold on;

% 标记前15个特征
scatter(top10_idx, top10_corr, 100, 'r', 'filled');

% 设置x轴标签
xlabel('特征', 'FontSize', 10);
ylabel('相关系数', 'FontSize', 10);
title('特征相关性分析 (红点表示前15个重要特征)', 'FontSize', 12, 'FontWeight', 'bold');

% 设置x轴刻度标签
set(gca, 'XTick', 1:n_features);
set(gca, 'XTickLabel', feature_names);
set(gca, 'XTickLabelRotation', 45);
set(gca, 'FontSize', 7);  % 缩小字体以适应长标签
grid on;
set(gca, 'LineWidth', 1.2);
xlim([0, n_features+1]);

% 调整布局
set(gcf, 'Color', 'w');

% 保存图片
saveas(gcf, '特征重要性分析.png');

