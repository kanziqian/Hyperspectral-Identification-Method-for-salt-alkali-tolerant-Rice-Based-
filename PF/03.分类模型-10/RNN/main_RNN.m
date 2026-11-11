% ---------------循环神经网络（RNN）识别分类模型（非线性）------------%

%% 清空环境变量
tic
clear ;       %清除所有变量
clc;
close all
disp('运行中...')

%% I. 导入数据
Data = xlsread('10个特征数据',1);   % 这行后面的1就是指sheet1
Data(1,:) = [];
Y = Data(:,end);
X = Data(:,1:end-1);    % 格式和X数据一样

rng(42);%种子数

%% II. 划分定标集和测试集(定标集80%/测试集20%)
X_train = [];
Y_train = [];
X_test = [];
Y_test = [];
for ii = 1:12
    id_T = find(Y==ii);
    Num_T = length(id_T);
    n_T = randperm(Num_T);
    X_T = X(id_T,:);
    Y_T = Y(id_T,:);
    nTrain_T = round(0.8*Num_T);   % 可改
    X_train = [X_train; X_T(n_T(1:nTrain_T),:)];
    Y_train = [Y_train; Y_T(n_T(1:nTrain_T),:)];

    X_test = [X_test; X_T(n_T(1+nTrain_T:end),:)];
    Y_test = [Y_test; Y_T(n_T(1+nTrain_T:end),:)];
end

% 注意：RNN需要特定的数据格式，保持原始格式
% X_train=X_train';
% Y_train=Y_train';
% 
% X_test=X_test';
% Y_test=Y_test';

%% III. RNN参数设置及数据预处理
% 1.参数设置***
numFeatures = size(X_train,2); % 特征数量
numHiddenUnits = 150; % 隐藏层单元数
numClasses = 12; % 输出类别数

% 2.数据预处理 - 转换为序列数据格式
% 将每个样本作为一个序列，序列长度为1，特征维度为numFeatures
X_train_cell = {};
Y_train_categorical = categorical(Y_train);

for i = 1:size(X_train,1)
    X_train_cell{i} = X_train(i,:)'; % 转置为列向量，格式：[特征数 × 序列长度]
end

X_test_cell = {};
Y_test_categorical = categorical(Y_test);
for i = 1:size(X_test,1)
    X_test_cell{i} = X_test(i,:)'; % 转置为列向量
end

% 数据归一化
% 计算训练集的均值和标准差用于归一化
mu = mean(X_train, 1);
sigma = std(X_train, 0, 1);

% 归一化训练集和测试集
X_train_normalized = (X_train - mu) ./ sigma;
X_test_normalized = (X_test - mu) ./ sigma;

% 更新归一化后的数据到cell数组
for i = 1:size(X_train_normalized,1)
    X_train_cell{i} = X_train_normalized(i,:)';
end
for i = 1:size(X_test_normalized,1)
    X_test_cell{i} = X_test_normalized(i,:)';
end

%% IV RNN训练及预测
% 1.构建RNN网络***
layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')
    
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm')
    
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'dropout1')
    
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% 2.训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 30, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% 3.RNN网络训练
net = trainNetwork(X_train_cell, Y_train_categorical, layers, options);

% 4.RNN网络预测
% -定标集
yfit_train_categorical = classify(net, X_train_cell);
yfit_train = double(yfit_train_categorical);

% -测试集
yfit_test_categorical = classify(net, X_test_cell);
yfit_test = double(yfit_test_categorical);

%%  数据排序（保持原代码格式）
[Y_train_sorted, index_1] = sort(Y_train);
[Y_test_sorted, index_2] = sort(Y_test);

yfit_train_sorted = yfit_train(index_1);
yfit_test_sorted = yfit_test(index_2);

%% V. 计算准确率及kappa系数
% 1.计算定标集和测试集整体准确率
train_rightnum = sum(yfit_train == Y_train)
train_Accuracy = train_rightnum./(length(Y_train)).*100
test_rightnum = sum(yfit_test == Y_test)
test_Accuracy = test_rightnum./(length(Y_test)).*100

% 2.计算不同类别预测准确率
acc_everyLabel_tr = [];
for ii = 1:12
    id_tr = find(Y_train == ii);
    if ~isempty(id_tr)
        Y_temp_real_tr = Y_train(id_tr);
        Y_temp_pred_tr = yfit_train(id_tr);
        train_rightnum1 = sum(Y_temp_pred_tr == Y_temp_real_tr);
        acc_everyLabel1 = train_rightnum1./(length(Y_temp_real_tr)).*100;
        acc_everyLabel_tr = [acc_everyLabel_tr; acc_everyLabel1];
    else
        acc_everyLabel_tr = [acc_everyLabel_tr; 0];
    end
end

acc_everyLabel_test = [];
for ii = 1:12
    id_test = find(Y_test == ii);
    if ~isempty(id_test)
        Y_temp_real_test = Y_test(id_test);
        Y_temp_pred_test = yfit_test(id_test);
        test_rightnum1 = sum(Y_temp_pred_test == Y_temp_real_test);
        acc_everyLabel2 = test_rightnum1./(length(Y_temp_real_test)).*100;
        acc_everyLabel_test = [acc_everyLabel_test; acc_everyLabel2];
    else
        acc_everyLabel_test = [acc_everyLabel_test; 0];
    end
end

% 3.计算kappa系数
kappa_train = KappaPara(Y_train, yfit_train)
kappa_test = KappaPara(Y_test, yfit_test)

%% VI. 绘图
FigureSave = fullfile('RNN结果图');   % 修改文件夹名称为RNN
mkdir(FigureSave)    % 创建文件夹
% 折线图
figure()
plot(1:length(Y_train_sorted),Y_train_sorted,'r-*',1:length(yfit_train_sorted),yfit_train_sorted,'b-o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('函数值')
string_1 = {'训练集预测结果对比';
    ['accuracy = ' num2str(train_Accuracy) '%']};
title(string_1)
savefig(gcf,fullfile(FigureSave ,'RNN训练集折线图'))    % 修改图名
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RNN训练集折线图'))

figure()
plot(1:length(Y_test_sorted),Y_test_sorted,'r-*',1:length(yfit_test_sorted),yfit_test_sorted,'b-o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('函数值')
string_2 = {'测试集预测结果对比';
    ['accuracy = ' num2str(test_Accuracy) '%']};
title(string_2)
savefig(gcf,fullfile(FigureSave ,'RNN测试集折线图'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RNN测试集折线图'))

% 混淆矩阵
figure
cm = confusionchart(Y_train, yfit_train);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'RNN训练集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RNN训练集混淆矩阵'))

figure
cm = confusionchart(Y_test, yfit_test);
cm.Title = 'Confusion Matrix for RNN Test Data';
%cm.ColumnSummary = 'column-normalized';
%cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'RNN测试集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RNN测试集混淆矩阵'))

