%-------------支持向量机（SVM）识别分类模型（非线性）------------------%

%% 清空环境变量
clear
close all
clc

%% I. 导入数据
Data = xlsread('10个特征数据',1);   % 这行后面的1就是指sheet1
%Data(1,:) = [];
Y = Data(:,end);
X = Data(:,1:end-1);

rng(42);%种子数


%% II. 划分定标集和测试集(定标集80%/测试集20%)
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

[c,g] = meshgrid(-10:0.5:10,-10:0.5:10);   %-10:0.5:10
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 6;   % 交叉验证次数
bestc = 1;
bestg = 0.1;
bestacc = 0;
% for i = 1:m
%     for j = 1:n
%         cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];
%         cg(i,j) = libsvmtrain(Y_train,Train_matrix,cmd);
%         if cg(i,j) > bestacc
%             bestacc = cg(i,j);
%             bestc = 2^c(i,j);
%             bestg = 2^g(i,j);
%         end
%         if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j)
%             bestacc = cg(i,j);
%             bestc = 2^c(i,j);
%             bestg = 2^g(i,j);
%         end
%     end
% end
% cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)]; % C和g
  % cmd = [' -t 2',' -c  1',' -g ,-2']; % C和g
  cmd = ['-t 2 -c 1 -g 1 -b 1']; 
  
% 2. 创建/训练SVM模型
model = libsvmtrain(Y_train,Train_matrix,cmd);   %libsvmtrain

% 3. SVM仿真测试
[predict_label_1,accuracy_1,prob_1] = libsvmpredict(Y_train,Train_matrix,model);  %定标集
[predict_label_2,accuracy_2,prob_2] = libsvmpredict(Y_test,Test_matrix,model);  %测试集

%% V. 计算准确率及kappa系数
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
    acc_everyLabel_tr = [acc_everyLabel_tr; acc_everyLabel1];   % 定标集单个分类的准确率
    acc_rightnum_tr = [acc_rightnum_tr; train_rightnum1];   % 定标集单个分类的准确个数
end
acc_everyLabel_test = [];
acc_rightnum_test = [];
for ii = 1:12   %****
    id_test = find(Y_test == ii);
    Y_temp_real_test = Y_test(id_test,:);
    Y_temp_pred_test = predict_label_2(id_test,:);
    test_rightnum1 = size(find(Y_temp_pred_test==Y_temp_real_test),1);
    acc_everyLabel2 = test_rightnum1./(size(Y_temp_real_test,1)).*100;
    acc_everyLabel_test = [acc_everyLabel_test; acc_everyLabel2];   % 测试集单个分类的准确率
    acc_rightnum_test = [acc_rightnum_test; test_rightnum1];   % 测试集单个分类的准确个数
end

% 3.计算kappa系数
kappa_train = KappaPara(Y_train, predict_label_1)
kappa_test = KappaPara(Y_test, predict_label_2)

%% VI. 绘图
% 折线图
FigureSave = fullfile('SVM结果图');   % 光谱处理图片保存至“结果图”文件夹中
mkdir(FigureSave)    % 创建文件夹

figure
plot(1:length(Y_train),Y_train,'r-*')
hold on
plot(1:length(Y_train),predict_label_1,'b-o')
grid on
legend('真实类别','预测类别','location','best','box','off')
xlabel('训练集样本编号')
ylabel('训练集样本类别')
string = {'训练集SVM预测结果对比(RBF核函数)';
    ['accuracy = ' num2str(accuracy_1(1)) '%']};
title(string)
savefig(gcf,fullfile(FigureSave ,'SVM训练集折线图'))    % 可保存为fig格式，方便后续图窗属性修改
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'SVM训练集折线图'))   % 保存为分辨率为600的PNG文件

figure
plot(1:length(Y_test),Y_test,'r-*')
hold on
plot(1:length(Y_test),predict_label_2,'b-o')
grid on
legend('真实类别','预测类别','location','best','box','off')
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集SVM预测结果对比(RBF核函数)';
    ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)
savefig(gcf,fullfile(FigureSave ,'SVM测试集折线图'))    % 可保存为fig格式，方便后续图窗属性修改
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'SVM测试集折线图'))   % 保存为分辨率为600的PNG文件

% 混淆矩阵
figure
cm = confusionchart(Y_train, predict_label_1);
cm.Title = 'Confusion Matrix for Train Data in SVM';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'SVM训练集混淆矩阵'))   % 可保存为fig格式，方便后续图窗属性修改
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'SVM训练集混淆矩阵'))   % 保存为分辨率为600的PNG文件

figure
cm = confusionchart(Y_test, predict_label_2);
cm.Title = 'Confusion Matrix for SVM Test Data ';
%cm.ColumnSummary = 'column-normalized';
%cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'SVM测试集混淆矩阵'))    % 可保存为fig格式，方便后续图窗属性修改
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'SVM测试集混淆矩阵'))   % 保存为分辨率为600的PNG文件



