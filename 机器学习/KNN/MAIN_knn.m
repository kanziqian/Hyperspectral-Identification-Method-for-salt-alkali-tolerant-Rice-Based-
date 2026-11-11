% ---------------K近邻（KNN）识别分类模型------------%

%% 清空环境变量
tic
clear ;       %清除所有变量
clc;
close all
disp('运行中...')

%% I. 导入数据
Data = xlsread('10+10个特征数据',1);   % 这行后面的1就是指sheet1
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

X_train=X_train';
Y_train=Y_train';

X_test=X_test';
Y_test=Y_test';

%% III. KNN参数设置及数据归一化处理
% 1.参数设置***
K = 3; % K值，可根据需要调整（常用3,5,7等奇数）
distance_metric = 'euclidean'; % 距离度量：'euclidean'(欧氏距离), 'cityblock'(曼哈顿距离)等


% 2.样本输入输出数据归一化（KNN对数据尺度敏感，需要归一化）
[inputn,inputps]=mapminmax(X_train);
% KNN不需要对输出进行特殊处理，保持原格式

%% IV KNN训练及预测
% 1.构建KNN模型***
% 使用fitcknn函数构建KNN分类器
knn_model = fitcknn(inputn', Y_train', ...
                   'NumNeighbors', K, ...
                   'Distance', distance_metric, ...
                   'Standardize', false); % 已经手动归一化，所以设为false

% 2.KNN模型预测
% -定标集
inputn_train = mapminmax('apply', X_train, inputps);
yfit_train = predict(knn_model, inputn_train')';

% -测试集
inputn_test = mapminmax('apply', X_test, inputps);
yfit_test = predict(knn_model, inputn_test')';

%%  数据排序（保持原代码不变）
[Y_train, index_1] = sort(Y_train);
[Y_test , index_2] = sort(Y_test);

yfit_train = yfit_train(index_1);
yfit_test = yfit_test(index_2);

%% V. 计算准确率及kappa系数
% 1.计算定标集和测试集整体准确率
train_rightnum = size(find(yfit_train==Y_train),2)
train_Accuracy = train_rightnum./(size(Y_train,2)).*100
test_rightnum = size(find(yfit_test==Y_test),2)
test_Accuracy = test_rightnum./(size(Y_test,2)).*100

% 2.计算不同类别预测准确率
acc_everyLabel_tr = [];
for ii = 1:12
    id_tr = find(Y_train == ii);
    Y_temp_real_tr = Y_train(:,id_tr);
    Y_temp_pred_tr = yfit_train(:,id_tr);
    train_rightnum1 = size(find(Y_temp_pred_tr==Y_temp_real_tr),2);
    acc_everyLabel1 = train_rightnum1./(size(Y_temp_real_tr,2)).*100;
    acc_everyLabel_tr = [acc_everyLabel_tr; acc_everyLabel1];
end
acc_everyLabel_test = [];
for ii = 1:12
    id_test = find(Y_test == ii);
    Y_temp_real_test = Y_test(:,id_test);
    Y_temp_pred_test = yfit_test(:,id_test);
    test_rightnum1 = size(find(Y_temp_pred_test==Y_temp_real_test),2);
    acc_everyLabel2 = test_rightnum1./(size(Y_temp_real_test,2)).*100;
    acc_everyLabel_test = [acc_everyLabel_test; acc_everyLabel2];
end

% 3.计算kappa系数
kappa_train = KappaPara(Y_train', yfit_train')
kappa_test = KappaPara(Y_test', yfit_test')

%% VI. 绘图
FigureSave = fullfile('KNN结果图');   % 修改文件夹名称为KNN
mkdir(FigureSave)    % 创建文件夹
% 折线图
figure()
plot(1:length(Y_train),Y_train,'r-*',1:length(yfit_train),yfit_train,'b-o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('函数值')
string_1 = {'训练集预测结果对比';
    ['accuracy = ' num2str(train_Accuracy) '%']};
title(string_1)
savefig(gcf,fullfile(FigureSave ,'KNN训练集折线图'))    % 修改图名
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'KNN训练集折线图'))

figure()
plot(1:length(Y_test),Y_test,'r-*',1:length(yfit_test),yfit_test,'b-o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('函数值')
string_2 = {'测试集预测结果对比';
    ['accuracy = ' num2str(test_Accuracy) '%']};
title(string_2)
savefig(gcf,fullfile(FigureSave ,'KNN测试集折线图'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'KNN测试集折线图'))

% 混淆矩阵
figure
cm = confusionchart(Y_train, yfit_train);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'KNN训练集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'KNN训练集混淆矩阵'))

figure
cm = confusionchart(Y_test, yfit_test);
cm.Title = 'Confusion Matrix for KNN Test Data';
%cm.ColumnSummary = 'column-normalized';
%cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'KNN测试集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'KNN测试集混淆矩阵'))
