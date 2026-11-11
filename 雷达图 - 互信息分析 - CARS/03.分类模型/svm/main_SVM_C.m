%-------------支持向量机（SVM）识别分类模型（非线性）------------------%

%% 清空环境变量
clear
close all
clc

%% I. 导入数据
Data = xlsread('CARS特征波段建模',1);   % 这行后面的1就是指sheet1
Data(1,:) = [];
Y = Data(:,end);
X = Data(:,1:end-1);

%% II. 划分定标集和测试集(定标集80%/测试集20%)
rng(42,'twister'); % 设置随机种子，保证结果可重复

X_train = [];
Y_train = [];
X_test = [];
Y_test = [];
for ii = 1:12   %****
    id_T = find(Y==ii);
    Num_T = length(id_T);
    n_T = randperm(Num_T);
    X_T = X(id_T,:);
    Y_T = Y(id_T,:);
    nTrain_T = round(0.8*Num_T);    % 可改
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

%% IV. SVM创建/训练(RBF核函数)惩罚参数C，核参数（g)
% 1. 寻找最佳c/g参数――网格交叉验证方法****
% （此处原网格搜索代码已注释，直接使用固定参数，如需要可取消注释）
cmd = ['-t 2 -c 1 -g 1 -b 1']; 
  
% 2. 创建/训练SVM模型
model = libsvmtrain(Y_train,Train_matrix,cmd);   %libsvmtrain

% 3. SVM仿真测试
[predict_label_1,accuracy_1,prob_1] = libsvmpredict(Y_train,Train_matrix,model);  %定标集
[predict_label_2,accuracy_2,prob_2] = libsvmpredict(Y_test,Test_matrix,model);  %测试集

%% V. 计算准确率、kappa系数、宏平均召回率及宏平均F1值
% 1.计算定标集和测试集整体准确率
train_rightnum = size(find(predict_label_1==Y_train),1)
train_Accuracy = train_rightnum./(size(Y_train,1)).*100
test_rightnum = size(find(predict_label_2==Y_test),1)
test_Accuracy = test_rightnum./(size(Y_test,1)).*100

% 2.计算不同类别预测准确率
acc_everyLabel_tr = [];
acc_rightnum_tr =[];
for ii = 1:12  %****
    id_tr = find(Y_train == ii);
    Y_temp_real_tr = Y_train(id_tr,:);
    Y_temp_pred_tr = predict_label_1(id_tr,:);
    train_rightnum1 = size(find(Y_temp_pred_tr==Y_temp_real_tr),1);
    acc_everyLabel1 = train_rightnum1./(size(Y_temp_real_tr,1)).*100;
    acc_everyLabel_tr = [acc_everyLabel_tr; acc_everyLabel1];
    acc_rightnum_tr = [acc_rightnum_tr; train_rightnum1];
end
acc_everyLabel_test = [];
acc_rightnum_test = [];
for ii = 1:12   %****
    id_test = find(Y_test == ii);
    Y_temp_real_test = Y_test(id_test,:);
    Y_temp_pred_test = predict_label_2(id_test,:);
    test_rightnum1 = size(find(Y_temp_pred_test==Y_temp_real_test),1);
    acc_everyLabel2 = test_rightnum1./(size(Y_temp_real_test,1)).*100;
    acc_everyLabel_test = [acc_everyLabel_test; acc_everyLabel2];
    acc_rightnum_test = [acc_rightnum_test; test_rightnum1];
end

% 3.计算kappa系数
kappa_train = KappaPara(Y_train, predict_label_1)
kappa_test = KappaPara(Y_test, predict_label_2)

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

%% VI. 绘图（全部改为中文宋体）
FigureSave = fullfile('SVM-CARS结果图');   % 光谱处理图片保存至“结果图”文件夹中
mkdir(FigureSave)    % 创建文件夹

% ---------- 训练集折线图 ----------
figure('Units','centimeters','Position',[10 5 12 8])
plot(1:length(Y_train),Y_train,'r-*','LineWidth',1,'MarkerSize',4)
hold on
plot(1:length(Y_train),predict_label_1,'b-o','LineWidth',1,'MarkerSize',4)
grid on
hleg = legend('真实类别','预测类别','Location','best','box','off');
set(hleg,'FontName','SimSun','FontSize',9)
xlabel('训练集样本编号','FontName','SimSun','FontSize',10)
ylabel('训练集样本类别','FontName','SimSun','FontSize',10)
string = {['训练集SVM预测结果对比(RBF核函数)']; ...
    ['准确率 = ' num2str(accuracy_1(1)) '%']};
title(string,'FontName','SimSun','FontSize',11)
set(gca,'FontName','Times New Roman','FontSize',9)
set(gcf,'Units','pixels')
savefig(gcf,fullfile(FigureSave ,'CARS训练集折线图'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'CARS训练集折线图'))

% ---------- 测试集折线图 ----------
figure('Units','centimeters','Position',[10 5 12 8])
plot(1:length(Y_test),Y_test,'r-*','LineWidth',1,'MarkerSize',4)
hold on
plot(1:length(Y_test),predict_label_2,'b-o','LineWidth',1,'MarkerSize',4)
grid on
hleg = legend('真实类别','预测类别','Location','best','box','off');
set(hleg,'FontName','SimSun','FontSize',9)
xlabel('测试集样本编号','FontName','SimSun','FontSize',10)
ylabel('测试集样本类别','FontName','SimSun','FontSize',10)
string = {['测试集SVM预测结果对比(RBF核函数)']; ...
    ['准确率 = ' num2str(accuracy_2(1)) '%']};
title(string,'FontName','SimSun','FontSize',11)
set(gca,'FontName','Times New Roman','FontSize',9)
set(gcf,'Units','pixels')
savefig(gcf,fullfile(FigureSave ,'CARS测试集折线图'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'CARS测试集折线图'))

% ---------- 训练集混淆矩阵 ----------
figure('Units','centimeters','Position',[10 5 12 10])
cm = confusionchart(Y_train, predict_label_1, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
cm.XLabel = '预测类别';
cm.YLabel = '真实类别';
cm.Title = 'SVM-CARS训练集混淆矩阵';
cm.FontName = 'SimSun';
cm.FontSize = 10;
set(gcf,'Units','pixels')
savefig(gcf,fullfile(FigureSave ,'CARS训练集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'CARS训练集混淆矩阵'))

% ---------- 测试集混淆矩阵 ----------
figure('Units','centimeters','Position',[10 5 12 10])
cm = confusionchart(Y_test, predict_label_2);
cm.XLabel = '预测类别';
cm.YLabel = '真实类别';
cm.Title = 'SVM-CARS测试集混淆矩阵';
cm.FontName = 'SimSun';
cm.FontSize = 10;
set(gcf,'Units','pixels')
savefig(gcf,fullfile(FigureSave ,'CARS测试集混淆矩阵'))
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'CARS测试集混淆矩阵'))