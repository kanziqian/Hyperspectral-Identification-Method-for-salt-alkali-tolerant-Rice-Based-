% ---------------偏最小二乘（PLSR）识别分类模型（线性）------------%

%% 清空环境变量
clear;
close all;
clc;

%% I. 导入数据
Data = xlsread('33个特征数据',1);   % 这行后面的1就是指sheet1
%Data(1,:) = [];   % 删掉波长数据
Y = Data(:,end);   % 标签
X = Data(:,1:end-1);    % 特征

%% II. 划分定标集（训练集）和测试集(定标集80%/测试集20%)
X_train = [];
Y_train = [];
X_test = [];
Y_test = [];
for ii = 1:12   %分为4类
    id_T = find(Y==ii);
    Num_T = length(id_T);
    n_T = randperm(Num_T);
    X_T = X(id_T,:);
    Y_T = Y(id_T,:);
    nTrain_T = round(0.8*Num_T);   % 可改成0.6/0.7

    X_train = [X_train; X_T(n_T(1:nTrain_T),:)];
    Y_train = [Y_train; Y_T(n_T(1:nTrain_T),:)];

    X_test = [X_test; X_T(n_T(1+nTrain_T:end),:)];
    Y_test = [Y_test; Y_T(n_T(1+nTrain_T:end),:)];

end

%% III. 偏最小二乘分析
% 求主成分数
train_XY=[X_train,Y_train];
data=zscore(train_XY); %数据标准化,变量记做 X*和 Y*
n=size(X_train,2); %n是自变量的个数
e0=data(:,1:end-1);f0=data(:,end); %标准化后的自变量和因变量数据
num=size(e0,1);%求样本点的个数
chg=eye(n); %w 到 w*变换矩阵的初始化
for i=1:n
    % 计算 w，w*和 t 的得分向量
    matrix=e0'*f0*f0'*e0;
    [vec,val]=eig(matrix); %求特征值和特征向量（[V,D]=eig(A)求矩阵A的全部特征值，构成对角阵D，并求A的特征向量构成V的列向量）
    val=diag(val); %提出对角线元素，即提出特征值
    [val,ind]=sort(val,'descend'); %对矩阵排序
    w(:,i)=vec(:,ind(1)); %提出最大特征值对应的特征向量
    w_star(:,i)=chg*w(:,i); %计算 w*的取值
    t(:,i)=e0*w(:,i); %计算成分 ti 的得分
    alpha=e0'*t(:,i)/(t(:,i)'*t(:,i)); %计算 alpha_i
    chg=chg*(eye(n)-w(:,i)*alpha'); %计算 w 到 w*的变换矩阵
    e=e0-t(:,i)*alpha'; %计算残差矩阵
    e0=e;
    % 计算 ss(i)的值
    beta=t\f0; %求回归方程的系数，数据标准化，没有常数项
    cancha1=f0-t*beta; %求残差矩阵
    ss(i)=sum(sum(cancha1.^2)); %求误差平方和
    % 计算 press(i)
    for j=1:num
        t1=t(:,1:i);f1=f0;
        she_t=t1(j,:);she_f=f1(j,:); %把舍去的第 j 个样本点保存起来
        t1(j,:)=[];f1(j,:)=[]; %删除第 j 个观测值
        beta1=[t1,ones(num-1,1)]\f1; %求回归分析的系数,这里带有常数项
        cancha2=she_f-she_t*beta1(1:end-1,:)-beta1(end,:); %求残差向量
        press_i(j)=sum(cancha2.^2); %求误差平方和
    end
    press(i)=sum(press_i);
    Q_h2(1)=1;
    if i>1, Q_h2(i)=1-press(i)/ss(i-1); end
    if Q_h2(i)<0.0975  % 交叉有效性大于0.0975****/==累计贡献率大于85%（论文）
        fprintf('提出的成分个数 r=%d',i); break
    end
end
Num_Comp=i;  %主成分个数

% PLSR分析
y_train=ind2vec(Y_train');
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(X_train,y_train',Num_Comp);

%% IV. 求主成分贡献率
fprintf('主成分贡献率百分比为：\n');
C_PC = 100*PCTVAR(2,:)'                % 求主成分贡献率
fprintf('主成分累积贡献率百分比为：\n');
Cumc_PC = cumsum(100*PCTVAR(2,:))'     % 求主成分累积贡献率

%% V. 计算拟合值
% -定标集
yfit_train = [ones(size(X_train,1),1) X_train]*beta;   % 计算拟合值
yfit_train = vec2ind(yfit_train');
yfit_train =yfit_train';

% -测试集
yfit_test = [ones(size(X_test,1),1) X_test]*beta;   % 计算拟合值
yfit_test = vec2ind(yfit_test');
yfit_test = yfit_test';

%% VI. 计算准确率及kappa系数
% 计算定标集和测试集整体准确率
train_rightnum = size(find(yfit_train==Y_train),1)
train_Accuracy = train_rightnum./(size(Y_train,1)).*100
test_rightnum = size(find(yfit_test==Y_test),1)
test_Accuracy = test_rightnum./(size(Y_test,1)).*100

% 计算不同分类预测准确率
acc_everyLabel_tr = [];
acc_rightnum_tr =[];
for ii = 1:12
    id_tr = find(Y_train == ii);
    Y_temp_real_tr = Y_train(id_tr,:);
    Y_temp_pred_tr = yfit_train(id_tr,:);
    train_rightnum1 = size(find(Y_temp_pred_tr==Y_temp_real_tr),1);
    acc_everyLabel1 = train_rightnum1./(size(Y_temp_real_tr,1)).*100;
    acc_everyLabel_tr = [acc_everyLabel_tr; acc_everyLabel1];   % 定标集单个分类的准确率
    acc_rightnum_tr = [acc_rightnum_tr; train_rightnum1];   % 定标集单个分类的准确个数
end
acc_everyLabel_test = [];
acc_rightnum_test = [];
for ii = 1:12
    id_test = find(Y_test == ii);
    Y_temp_real_test = Y_test(id_test,:);
    Y_temp_pred_test = yfit_test(id_test,:);
    test_rightnum1 = size(find(Y_temp_pred_test==Y_temp_real_test),1);
    acc_everyLabel2 = test_rightnum1./(size(Y_temp_real_test,1)).*100;
    acc_everyLabel_test = [acc_everyLabel_test; acc_everyLabel2];   % 测试集单个分类的准确率
    acc_rightnum_test = [acc_rightnum_test; test_rightnum1];   % 测试集单个分类的准确个数
end

% 计算kappa系数
kappa_train = KappaPara(Y_train, yfit_train)
kappa_test = KappaPara(Y_test, yfit_test)

%% VII. 绘图
FigureSave = fullfile('PLSR结果图');   % 光谱处理图片保存至“结果图”文件夹中
mkdir(FigureSave)    % 创建文件夹
% 主成分贡献率
figure()
plot(1:Num_Comp,Cumc_PC,'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

% 折线图
figure()
plot(1:length(Y_train),Y_train,'r-*',1:length(yfit_train),yfit_train,'b-o')
grid on
legend('真实类别','预测类别','location','northwest','box','off')
xlabel('样本编号')
ylabel('函数值')
string_1 = {'定标集预测结果对比';
    ['accuracy = ' num2str(train_Accuracy) '%']};
title(string_1)
savefig(gcf,fullfile(FigureSave ,'PLSR训练集折线图'))    % 可保存为fig格式，方便后续图窗属性修改
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'PLSR训练集折线图'))   % 保存为分辨率为600的PNG文件

figure()
plot(1:length(Y_test),Y_test,'r-*',1:length(yfit_test),yfit_test,'b-o')
grid on
legend('真实类别','预测类别','location','northwest','box','off')
xlabel('样本编号')
ylabel('函数值')
string_2 = {'测试集预测结果对比';
    ['accuracy = ' num2str(test_Accuracy) '%']};
title(string_2)
savefig(gcf,fullfile(FigureSave ,'PLSR测试集折线图'))    % 可保存为fig格式，方便后续图窗属性修改
print(gcf,'-dpng','-r600',fullfile(FigureSave ,strcat('PLSR测试集折线图')))   % 保存为分辨率为600的PNG文件

% 混淆矩阵
figure
cm = confusionchart(Y_train, yfit_train);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'PLSR训练集混淆矩阵'))    % 可保存为fig格式，方便后续图窗属性修改
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'PLSR训练集混淆矩阵'))   % 保存为分辨率为600的PNG文件

figure
cm = confusionchart(Y_test, yfit_test);
cm.Title = 'Confusion Matrix for PLSR Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
savefig(gcf,fullfile(FigureSave ,'PLSR测试集混淆矩阵'))    % 可保存为fig格式，方便后续图窗属性修改
print(gcf,'-dpng','-r600',fullfile(FigureSave ,'PLSR测试集混淆矩阵'))   % 保存为分辨率为600的PNG文件



