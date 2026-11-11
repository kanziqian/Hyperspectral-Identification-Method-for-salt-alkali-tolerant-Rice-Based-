% ---------------随机森林（RF）识别分类模型（非线性）------------%

%% 清空环境变量
tic
clear ;       %清除所有变量
clc;
close all
disp('运行中...')

%% I. 导入数据
Data = xlsread('10个特征数据',1);   % 这行后面的1就是指sheet1
%Data(1,:) = [];
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

% 注意：RF不需要转置，保持原始格式（样本×特征）
% X_train=X_train';
% Y_train=Y_train';
% 
% X_test=X_test';
% Y_test=Y_test';

%% III. 随机森林参数设置及数据预处理
% 1.参数设置***
numTrees = 40; % 树的数量，可根据需要调整
minLeafSize = 2; % 最小叶子大小，防止过拟合


% 2.数据预处理（RF对数据尺度不敏感，但保持归一化步骤）
% 转换为Table格式，TreeBagger需要
train_data = array2table(X_train);
test_data = array2table(X_test);

% 添加响应变量名
for i = 1:size(X_train,2)
    train_data.Properties.VariableNames{i} = ['Feature_' num2str(i)];
    test_data.Properties.VariableNames{i} = ['Feature_' num2str(i)];
end

%% IV 随机森林训练及预测
% 1.构建随机森林模型***
rf_model = TreeBagger(numTrees, train_data, Y_train, ...
                     'Method', 'classification', ...
                     'MinLeafSize', minLeafSize, ...
                     'OOBPrediction', 'on', ...
                     'OOBPredictorImportance', 'on');

% 2.模型预测
% -定标集
[yfit_train, train_scores] = predict(rf_model, train_data);
yfit_train = str2double(yfit_train);

% -测试集
[yfit_test, test_scores] = predict(rf_model, test_data);
yfit_test = str2double(yfit_test);

%%  数据排序（保持原代码不变）
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
    Y_temp_real_tr = Y_train(id_tr);
    Y_temp_pred_tr = yfit_train(id_tr);
    train_rightnum1 = sum(Y_temp_pred_tr == Y_temp_real_tr);
    acc_everyLabel1 = train_rightnum1./(length(Y_temp_real_tr)).*100;
    acc_everyLabel_tr = [acc_everyLabel_tr; acc_everyLabel1];
end
acc_everyLabel_test = [];
for ii = 1:12
    id_test = find(Y_test == ii);
    Y_temp_real_test = Y_test(id_test);
    Y_temp_pred_test = yfit_test(id_test);
    test_rightnum1 = sum(Y_temp_pred_test == Y_temp_real_test);
    acc_everyLabel2 = test_rightnum1./(length(Y_temp_real_test)).*100;
    acc_everyLabel_test = [acc_everyLabel_test; acc_everyLabel2];
end

% 3.计算kappa系数
kappa_train = KappaPara(Y_train, yfit_train)
kappa_test = KappaPara(Y_test, yfit_test)

%% VI. 绘图
FigureSave = fullfile('RF结果图');   % 修改文件夹名称为RF
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
savefig(gcf,fullfile(FigureSave ,'RF训练集折线图'))    % 修改图名
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RF训练集折线图'))

figure()
plot(1:length(Y_test_sorted),Y_test_sorted,'r-*',1:length(yfit_test_sorted),yfit_test_sorted,'b-o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('函数值')
string_2 = {'测试集预测结果对比';
    ['accuracy = ' num2str(test_Accuracy) '%']};
title(string_2)
savefig(gcf,fullfile(FigureSave ,'RF测试集折线图'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RF测试集折线图'))

% 混淆矩阵
figure
cm = confusionchart(Y_train, yfit_train);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'RF训练集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RF训练集混淆矩阵'))

figure
cm = confusionchart(Y_test, yfit_test);
cm.Title = 'Confusion Matrix for RF Test Data';
%cm.ColumnSummary = 'column-normalized';
%cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'RF测试集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RF测试集混淆矩阵'))

% 新增：特征重要性图
figure
importance = rf_model.OOBPermutedPredictorDeltaError;
bar(importance)
xlabel('特征编号')
ylabel('重要性得分')
title('随机森林特征重要性')
savefig(gcf,fullfile(FigureSave ,'RF特征重要性'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'RF特征重要性'))