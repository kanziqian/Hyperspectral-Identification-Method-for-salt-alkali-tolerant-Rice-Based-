function kappa = KappaPara(Y_real, Y_pred)
% 计算kappa系数
    A=[Y_pred, Y_real];
    class=intersect(Y_real,Y_real);
    num_class=length(class);
    [m,~]=size(A);
    result=zeros(num_class,num_class);%初始化结果数据
    for k=1:m
        for i=1:num_class
            for j=1:num_class
                if sum(A(k,:)==[class(i),(j)])==2
                    result(i,j)=result(i,j)+1;
                end
            end
        end
    end
    p0=trace(result)/m;
    pe=sum(result,1)*sum(result,2)/(m^2);
    kappa=(p0-pe)/(1-pe);

end