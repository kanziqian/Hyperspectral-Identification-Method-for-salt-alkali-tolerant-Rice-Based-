function [macro_recall, macro_f1, recall, precision, f1] = calculate_macro_metrics(true_labels, pred_labels, num_classes)
% 计算宏平均召回率和宏平均F1值
% 输入:
%   true_labels - 真实标签
%   pred_labels - 预测标签
%   num_classes - 类别数量
% 输出:
%   macro_recall - 宏平均召回率
%   macro_f1 - 宏平均F1值
%   recall - 每个类别的召回率
%   precision - 每个类别的精确率
%   f1 - 每个类别的F1值

    recall = zeros(num_classes, 1);
    precision = zeros(num_classes, 1);
    f1 = zeros(num_classes, 1);
    
    for ii = 1:num_classes
        % 真正例 (True Positive)
        TP = sum((true_labels == ii) & (pred_labels == ii));
        % 假正例 (False Positive)
        FP = sum((true_labels ~= ii) & (pred_labels == ii));
        % 假负例 (False Negative)
        FN = sum((true_labels == ii) & (pred_labels ~= ii));
        
        % 计算精确率
        if (TP + FP) > 0
            precision(ii) = TP / (TP + FP);
        else
            precision(ii) = 0;
        end
        
        % 计算召回率
        if (TP + FN) > 0
            recall(ii) = TP / (TP + FN);
        else
            recall(ii) = 0;
        end
        
        % 计算F1值
        if (precision(ii) + recall(ii)) > 0
            f1(ii) = 2 * precision(ii) * recall(ii) / (precision(ii) + recall(ii));
        else
            f1(ii) = 0;
        end
    end
    
    % 计算宏平均
    macro_recall = mean(recall);
    macro_f1 = mean(f1);
end