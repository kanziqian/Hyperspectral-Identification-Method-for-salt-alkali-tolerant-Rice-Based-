%-------------K最近邻（KNN）识别分类模型------------------%

%% 清空环境变量
clear
close all
clc

%% I. 导入数据
Data = xlsread('CARS特征波段建模.xls',1);   % 这行后面的1就是指sheet1
Data(1,:) = [];
Y = Data(:,end);
X = Data(:,1:end-1);

%% II. 划分训练集和测试集(训练集80%/测试集20%)
rng(42,'twister'); % 设置随机种子，保证结果可重复

X_train = [];
Y_train = [];
X_test = [];
Y_test = [];
for ii = 1:12   % 12个类别
    id_T = find(Y==ii);
    Num_T = length(id_T);
    n_T = randperm(Num_T);
    X_T = X(id_T,:);
    Y_T = Y(id_T,:);
    nTrain_T = round(0.8*Num_T);    % 训练集比例
    X_train = [X_train; X_T(n_T(1:nTrain_T),:)];
    Y_train = [Y_train; Y_T(n_T(1:nTrain_T),:)];

    X_test = [X_test; X_T(n_T(1+nTrain_T:end),:)];
    Y_test = [Y_test; Y_T(n_T(1+nTrain_T:end),:)];
end

%% III. 数据归一化
[Train_matrix,PS] = mapminmax(X_train');
Train_matrix = Train_matrix';
Test_matrix = mapminmax('apply',X_test',PS);
Test_matrix = Test_matrix';

%% IV. KNN创建/训练
% 1. 寻找最佳K值――交叉验证方法
fprintf('正在寻找最佳K值...\n');
max_k = 15; % 最大K值搜索范围
cv_acc = zeros(max_k, 1);

for k = 1:max_k
    % 创建KNN模型
    knn_model = fitcknn(Train_matrix, Y_train, 'NumNeighbors', k, 'Standardize', false);
    
    % 交叉验证
    cv_model = crossval(knn_model, 'KFold', 5);
    cv_acc(k) = 1 - kfoldLoss(cv_model);
end

% 找到最佳K值
[best_acc, best_k] = max(cv_acc);
fprintf('最佳K值: %d, 交叉验证准确率: %.2f%%\n', best_k, best_acc*100);

% 绘制K值选择曲线
figure
plot(1:max_k, cv_acc*100, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on
plot(best_k, best_acc*100, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('K值');
ylabel('交叉验证准确率(%)');
title('K值选择曲线');
legend('准确率', '最佳K值', 'Location', 'best');
grid on;

% 2. 使用最佳K值创建最终KNN模型
train_start = tic;
knn_final = fitcknn(Train_matrix, Y_train, ...
                   'NumNeighbors', best_k, ...
                   'Standardize', false, ...
                   'Distance', 'euclidean'); % 可以使用其他距离度量：'cityblock', 'cosine', 'correlation'

train_elapsed = toc(train_start);
fprintf('KNN模型构建（训练）时间：%.4f 秒 (%.2f 毫秒)\n', train_elapsed, train_elapsed*1000);
% 3. KNN预测
predict_label_1 = predict(knn_final, Train_matrix);  % 训练集
predict_label_2 = predict(knn_final, Test_matrix);   % 测试集

%% V. 计算准确率及kappa系数
% 1.计算训练集和测试集整体准确率
train_rightnum = sum(predict_label_1 == Y_train);
train_Accuracy = train_rightnum / length(Y_train) * 100;
test_rightnum = sum(predict_label_2 == Y_test);
test_Accuracy = test_rightnum / length(Y_test) * 100;

fprintf('训练集准确率: %.2f%% (%d/%d)\n', train_Accuracy, train_rightnum, length(Y_train));
fprintf('测试集准确率: %.2f%% (%d/%d)\n', test_Accuracy, test_rightnum, length(Y_test));

% 2.计算不同类别预测准确率
acc_everyLabel_tr = [];
acc_rightnum_tr =[];
for ii = 1:12
    id_tr = find(Y_train == ii);
    Y_temp_real_tr = Y_train(id_tr,:);
    Y_temp_pred_tr = predict_label_1(id_tr,:);
    train_rightnum1 = sum(Y_temp_pred_tr == Y_temp_real_tr);
    acc_everyLabel1 = train_rightnum1 / length(Y_temp_real_tr) * 100;
    acc_everyLabel_tr = [acc_everyLabel_tr; acc_everyLabel1];
    acc_rightnum_tr = [acc_rightnum_tr; train_rightnum1];
end

acc_everyLabel_test = [];
acc_rightnum_test = [];
for ii = 1:12
    id_test = find(Y_test == ii);
    Y_temp_real_test = Y_test(id_test,:);
    Y_temp_pred_test = predict_label_2(id_test,:);
    test_rightnum1 = sum(Y_temp_pred_test == Y_temp_real_test);
    acc_everyLabel2 = test_rightnum1 / length(Y_temp_real_test) * 100;
    acc_everyLabel_test = [acc_everyLabel_test; acc_everyLabel2];
    acc_rightnum_test = [acc_rightnum_test; test_rightnum1];
end

% 显示每个类别的准确率
fprintf('\n训练集各类别准确率:\n');
for ii = 1:12
    fprintf('类别 %2d: %.2f%% (%d/%d)\n', ii, acc_everyLabel_tr(ii), acc_rightnum_tr(ii), sum(Y_train==ii));
end

fprintf('\n测试集各类别准确率:\n');
for ii = 1:12
    fprintf('类别 %2d: %.2f%% (%d/%d)\n', ii, acc_everyLabel_test(ii), acc_rightnum_test(ii), sum(Y_test==ii));
end

% 3.计算kappa系数
kappa_train = KappaPara(Y_train, predict_label_1);
kappa_test = KappaPara(Y_test, predict_label_2);

fprintf('\nKappa系数:\n');
fprintf('训练集Kappa: %.4f\n', kappa_train);
fprintf('测试集Kappa: %.4f\n', kappa_test);

% 4.计算训练集的宏平均召回率和宏平均F1值
[macro_recall_train, macro_f1_train, recall_train, precision_train, f1_train] = calculate_macro_metrics(Y_train, predict_label_1, 12);
fprintf('训练集宏平均召回率: %.2f%%\n', macro_recall_train*100);
fprintf('训练集宏平均F1值: %.2f%%\n', macro_f1_train*100);

% 5.计算测试集的宏平均召回率和宏平均F1值
[macro_recall_test, macro_f1_test, recall_test, precision_test, f1_test] = calculate_macro_metrics(Y_test, predict_label_2, 12);
fprintf('测试集宏平均召回率: %.2f%%\n', macro_recall_test*100);
fprintf('测试集宏平均F1值: %.2f%%\n', macro_f1_test*100);

% 6.显示每个类别的详细指标
fprintf('\n=== 训练集各类别详细指标 ===\n');
for ii = 1:12
    fprintf('类别 %d: 精确率=%.2f%%, 召回率=%.2f%%, F1值=%.2f%%\n', ...
        ii, precision_train(ii)*100, recall_train(ii)*100, f1_train(ii)*100);
end
fprintf('\n=== 测试集各类别详细指标 ===\n');
for ii = 1:12
    fprintf('类别 %d: 精确率=%.2f%%, 召回率=%.2f%%, F1值=%.2f%%\n', ...
        ii, precision_test(ii)*100, recall_test(ii)*100, f1_test(ii)*100);
end

%% VI. 绘图
% 创建保存结果的文件夹
FigureSave = fullfile('KNN-CARS结果图');
mkdir(FigureSave);

% 折线图
figure
plot(1:length(Y_train), Y_train, 'r-*', 'LineWidth', 1, 'MarkerSize', 4)
hold on
plot(1:length(Y_train), predict_label_1, 'b-o', 'LineWidth', 1, 'MarkerSize', 4)
grid on
legend('真实类别', '预测类别', 'location', 'best', 'box', 'off')
xlabel('训练集样本编号')
ylabel('训练集样本类别')
string = {'训练集KNN预测结果对比';
    ['accuracy = ' num2str(train_Accuracy, '%.2f') '%'];
    ['K = ' num2str(best_k)]};
title(string)
savefig(gcf, fullfile(FigureSave, 'KNN训练集折线图'))
print(gcf, '-dpng', '-r600', fullfile(FigureSave, 'KNN训练集折线图'))

figure
plot(1:length(Y_test), Y_test, 'r-*', 'LineWidth', 1, 'MarkerSize', 4)
hold on
plot(1:length(Y_test), predict_label_2, 'b-o', 'LineWidth', 1, 'MarkerSize', 4)
grid on
legend('真实类别', '预测类别', 'location', 'best', 'box', 'off')
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集KNN预测结果对比';
    ['accuracy = ' num2str(test_Accuracy, '%.2f') '%'];
    ['K = ' num2str(best_k)]};
title(string)
savefig(gcf, fullfile(FigureSave, 'KNN测试集折线图'))
print(gcf, '-dpng', '-r600', fullfile(FigureSave, 'KNN测试集折线图'))

% 混淆矩阵
figure
cm = confusionchart(Y_train, predict_label_1);
xlabel('Predicted Classes');
ylabel('Ture Classes');
cm.Title = 'Confusion Matrix for Train Data in KNN-CARS';
savefig(gcf, fullfile(FigureSave, 'KNN训练集混淆矩阵'))
print(gcf, '-dpng', '-r600', fullfile(FigureSave, 'KNN训练集混淆矩阵'))

figure
cm = confusionchart(Y_test, predict_label_2);
xlabel('Predicted Classes');
ylabel('Ture Classes');
cm.Title = 'Confusion Matrix for Test Data in KNN-CARS';
savefig(gcf, fullfile(FigureSave, 'KNN测试集混淆矩阵'))
print(gcf, '-dpng', '-r600', fullfile(FigureSave, 'KNN测试集混淆矩阵'))
