%-----------Feature Correlation Analysis--------------
%% Clear Environment Variables
clc; close all; clear; warning off;

%% Import Data
x = xlsread('CARS特征波段建模.xls', 1);
data = x(2:end, 1:end);

n_features = 39;     % Number of features

% -----------------------------------------------------------------------------
data_normalized = zeros(size(data, 1), n_features); % Pre-allocate memory

for i = 1:n_features
    % Calculate mean and standard deviation
    mu = mean(data(:, i));
    sigma = std(data(:, i));
    
    % Z-score normalization
    data_normalized(:, i) = (data(:, i) - mu) / sigma;
end

features = data_normalized(:, 1:end);
y = data(:, end);

%% Correlation Analysis
features_corr = corr(features, y);

%% Get Top 15 Most Correlated Features (adjustable)
[sorted_corr, sorted_idx] = sort(abs(features_corr), 'descend');
top10_idx = sorted_idx(1:min(15, length(sorted_idx))); %%%
top10_corr = features_corr(top10_idx);

%% Save Results to Excel
result_table = table((1:length(top10_idx))', top10_idx, top10_corr, ...
    'VariableNames', {'Rank', 'Feature_Index', 'Correlation_Coefficient'});
writetable(result_table, 'Top15_Correlated_Features.xlsx');

%% Visualization - Create Aesthetic Figure
figure('Position', [100, 100, 1400, 800], 'Color', [0.95, 0.95, 0.95], ...
    'Name', 'Feature Correlation Analysis');

% Create English feature names with proper capitalization
feature_names = cell(1, 39);
feature_names{1} = '462nm';
feature_names{2} = '463nm';
feature_names{3} = '488nm';
feature_names{4} = '492nm';
feature_names{5} = '509nm';
feature_names{6} = '510nm';
feature_names{7} = '593nm';
feature_names{8} = '594nm';
feature_names{9} = '595nm';
feature_names{10} = '596nm';
feature_names{11} = '635nm';
feature_names{12} = '636nm';
feature_names{13} = '637nm';
feature_names{14} = '638nm';
feature_names{15} = '690nm';
feature_names{16} = '691nm';
feature_names{17} = '718nm';
feature_names{18} = '750nm';
feature_names{19} = '751nm';
feature_names{20} = '942nm';
feature_names{21} = '961nm';
feature_names{22} = '994nm';
feature_names{23} = '995nm';
feature_names{24} = '1025nm';
feature_names{25} = '1403nm';
feature_names{26} = '1404nm';
feature_names{27} = '1409nm';
feature_names{28} = '1414nm';
feature_names{29} = '1415nm';
feature_names{30} = '1416nm';
feature_names{31} = '1570nm';
feature_names{32} = '1576nm';
feature_names{33} = '1622nm';
feature_names{34} = '1692nm';
feature_names{35} = '1693nm';
feature_names{36} = '1694nm';
feature_names{37} = '1695nm';
feature_names{38} = '2229nm';
feature_names{39} = '2270nm';


% Create subplot layout
subplot(2, 1, 1);

% Plot bar chart with gradient color
bar_colors = parula(n_features);  % Use color gradient
for i = 1:n_features
    bar(i, features_corr(i), 'FaceColor', bar_colors(i, :), ...
        'EdgeColor', 'k', 'LineWidth', 0.5);
    hold on;
end

% Mark top 15 features with red markers
scatter(top10_idx, top10_corr, 120, 'r', 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

% Customize plot appearance
title(' ', ...                                     %标题头
    'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.1, 0.1, 0.1]);
% 第一个子图（柱状图）
xlabel('Features Number', 'FontName', 'Calibri', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('MRMR Score',      'FontName', 'Calibri', 'FontSize', 20, 'FontWeight', 'bold');

% Set x-axis properties
set(gca, 'XTick', 1:n_features, 'XTickLabel', feature_names, ...
    'XTickLabelRotation', 45, 'FontSize', 9, ...
    'GridColor', [0.7, 0.7, 0.7], 'GridAlpha', 0.5, ...
    'LineWidth', 1.5, 'Box', 'on');

grid on;
grid minor;
xlim([0, n_features+1]);
ylim([min(features_corr)-0.1, max(features_corr)+0.1]);



% Subplot 2: Top 15 Features Horizontal Bar Chart
subplot(2, 1, 2);

% Sort top features for better visualization
[top_corr_sorted, sort_idx] = sort(top10_corr, 'ascend');
top_idx_sorted = top10_idx(sort_idx);

% Create horizontal bar chart
barh(1:length(top_idx_sorted), top_corr_sorted, 'FaceColor', [0.2, 0.6, 0.8], ...
    'EdgeColor', 'k', 'LineWidth', 1);

% Customize appearance
title('Top 15 Correlated Features', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.1, 0.1, 0.1]);
xlabel('Correlation Coefficient', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Feature Index', 'FontSize', 12, 'FontWeight', 'bold');

% Add feature names as y-tick labels
set(gca, 'YTick', 1:length(top_idx_sorted), ...
    'YTickLabel', feature_names(top_idx_sorted), ...
    'FontSize', 10, 'LineWidth', 1.5, 'Box', 'on');

grid on;
grid minor;
xlim([min(top_corr_sorted)-0.1, max(top_corr_sorted)+0.1]);
ylim([0, length(top_idx_sorted)+1]);

% Add correlation coefficient values on bars
for i = 1:length(top_corr_sorted)
    if top_corr_sorted(i) >= 0
        text_x = top_corr_sorted(i) + 0.01;
        text_align = 'left';
    else
        text_x = top_corr_sorted(i) - 0.01;
        text_align = 'right';
    end
    text(text_x, i, sprintf('%.3f', top_corr_sorted(i)), ...
        'FontSize', 9, 'FontWeight', 'bold', ...
        'VerticalAlignment', 'middle', 'HorizontalAlignment', text_align);
end

% Adjust layout
set(gcf, 'Color', 'w');
tight_layout = get(gca, 'TightInset');
set(gca, 'Position', [tight_layout(1), tight_layout(2), ...
    0.95-tight_layout(1)-tight_layout(3), 0.45-tight_layout(2)-tight_layout(4)]);

% Save high-quality figure
exportgraphics(gcf, 'Feature_Correlation_Analysis.png', 'Resolution', 600);

% Display summary in command window
fprintf('\n========== FEATURE CORRELATION ANALYSIS ==========\n');
fprintf('Total Features Analyzed: %d\n', n_features);
fprintf('Top 5 Most Correlated Features:\n');
for i = 1:min(5, length(top10_idx))
    fprintf('  %d. Feature %d: %s (Correlation: %.4f)\n', ...
        i, top10_idx(i), feature_names{top10_idx(i)}, top10_corr(i));
end
fprintf('\nAnalysis completed successfully!\n');
fprintf('Results saved to:\n');
fprintf('  1. Feature_Correlation_Analysis.png\n');
fprintf('  2. Top20_Correlated_Features.xlsx\n');

