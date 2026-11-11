% ---------------BP神经网络（BPNN）识别分类模型（非线性）------------% 

%% 清空环境变量
tic
clear ;       %清除所有变量
clc;
close all
disp('运行中...')

%% I. 导入数据
Data = xlsread('CARS特征波段建模.xls',1);   % 这行后面的1就是指sheet1
Data(1,:) = [];
Y = Data(:,end);
X = Data(:,1:end-1);    % 格式和X数据一样

%% II. 划分定标集和测试集(定标集80%/测试集20%)
rng(42,'twister'); % 设置随机种子，保证结果可重复

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

%% III. BP神经网络参数设置及数据归一化处理
% 1.参数设置***
inputnum=size(X_train,1);%输入节点数  特征数量
hiddennum=2*inputnum+1;%中间节点数   *****隐含层节点数 经验公式：隐含层节点数=2*输入层节点数+1
outputnum=12;%输出节点数

% 2.样本输入输出数据归一化
[inputn,inputps]=mapminmax(X_train);
outputn = ind2vec(Y_train);

%% IV BP神经网络训练及预测
% 1.构建BP网络***
net=newff(inputn,outputn,hiddennum);    % 构建模型的函数newff
net.trainParam.epochs=100;   % 迭代次数
net.trainParam.lr=0.01;   %  学习率
net.trainParam.goal=0.000001;   % 损失函数的目标值

net.trainFcn= 'trainscg';
% 2.BP网络训练
[net,per2]=train(net,inputn,outputn);

% 3.BP网络预测
% -定标集
inputn_train=mapminmax('apply',X_train,inputps);
an_train=sim(net,inputn_train);

% -测试集
inputn_test=mapminmax('apply',X_test,inputps);
an_test=sim(net,inputn_test);

%%  数据反归一化
yfit_train = vec2ind(an_train);
yfit_test = vec2ind(an_test);

%%  数据排序
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

% 4.计算训练集的宏平均召回率和宏平均F1值
[macro_recall_train, macro_f1_train, recall_train, precision_train, f1_train] = calculate_macro_metrics(Y_train, yfit_train, 12);
fprintf('训练集宏平均召回率: %.2f%%\n', macro_recall_train*100);
fprintf('训练集宏平均F1值: %.2f%%\n', macro_f1_train*100);

% 5.计算测试集的宏平均召回率和宏平均F1值
[macro_recall_test, macro_f1_test, recall_test, precision_test, f1_test] = calculate_macro_metrics(Y_test, yfit_test, 12);
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

%% VI. 绘图（全部改为中文宋体）
FigureSave = fullfile('BPNN结果图');   % 光谱处理图片保存至“结果图”文件夹中
mkdir(FigureSave)    % 创建文件夹

% ---------- 训练集折线图 ----------
figure('Units','centimeters','Position',[10 5 12 8])
plot(1:length(Y_train),Y_train,'r-*',1:length(yfit_train),yfit_train,'b-o','LineWidth',1.2)
grid on
hleg = legend('真实值','预测值','Location','best');
set(hleg,'FontName','SimSun','FontSize',9)
xlabel('样本编号','FontName','SimSun','FontSize',10)
ylabel('类别标签','FontName','SimSun','FontSize',10)
string_1 = {['训练集预测结果对比'];['准确率 = ' num2str(train_Accuracy,'%.2f') '%']};
title(string_1,'FontName','SimSun','FontSize',11)
set(gca,'FontName','Times New Roman','FontSize',9)
% 打印前恢复单位
set(gcf,'Units','pixels')
savefig(gcf,fullfile(FigureSave ,'CARS训练集折线图'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'CARS训练集折线图'))

% ---------- 测试集折线图 ----------
figure('Units','centimeters','Position',[10 5 12 8])
plot(1:length(Y_test),Y_test,'r-*',1:length(yfit_test),yfit_test,'b-o','LineWidth',1.2)
grid on
hleg = legend('真实值','预测值','Location','best');
set(hleg,'FontName','SimSun','FontSize',9)
xlabel('样本编号','FontName','SimSun','FontSize',10)
ylabel('类别标签','FontName','SimSun','FontSize',10)
string_2 = {['测试集预测结果对比'];['准确率 = ' num2str(test_Accuracy,'%.2f') '%']};
title(string_2,'FontName','SimSun','FontSize',11)
set(gca,'FontName','Times New Roman','FontSize',9)
set(gcf,'Units','pixels')
savefig(gcf,fullfile(FigureSave ,'CARS测试集折线图'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'CARS测试集折线图'))

% ---------- 训练集混淆矩阵 ----------
figure('Units','centimeters','Position',[10 5 12 10])
cm = confusionchart(Y_train, yfit_train, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
% 修改标签和标题为中文
cm.XLabel = '预测类别';
cm.YLabel = '真实类别';
cm.Title = 'BPNN-CARS训练集混淆矩阵';
% 设置字体
cm.FontName = 'SimSun';
cm.FontSize = 10;
% 如需单独设置轴标签字体大小，可用以下方式：
% cm.XLabelFontSize = 10; cm.YLabelFontSize = 10;
set(gcf,'Units','pixels')
savefig(gcf,fullfile(FigureSave ,'CARS训练集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'CARS训练集混淆矩阵'))

% ---------- 测试集混淆矩阵 ----------
figure('Units','centimeters','Position',[10 5 12 10])
cm = confusionchart(Y_test, yfit_test);
cm.XLabel = '预测类别';
cm.YLabel = '真实类别';
cm.Title = 'BPNN-CARS测试集混淆矩阵';
cm.FontName = 'SimSun';
cm.FontSize = 10;
set(gcf,'Units','pixels')
savefig(gcf,fullfile(FigureSave ,'CARS测试集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'CARS测试集混淆矩阵'))

toc