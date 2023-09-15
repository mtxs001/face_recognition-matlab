clc;clear;close all;
load('ORL_32x32.mat');
%% 训练集测试集
Fea=zeros(400,32);
for i=1:400
    a=zeros(32,32);   
    a(:)=fea(i,:);
    s=svd(cov(a));
    b=a*s;
    Fea(i,:)=b;
end
p=0.3;
fea=zscore(Fea);
[Train,Test]=crossvalind('Holdout',gnd,p);
[a,~]=size(gnd);
a=zeros(a,a/10);
for i=1:400    
    a(i,gnd(i))=1;
end
gnd1=a;
input_train=fea(Train,:);
output_train=gnd1(Train,:);
input_test=fea(Test,:);
output_test=gnd(Test,:);
%% BP神 经 网 络
bp2 = nnsetup([32 200 40]);%1024个 输 入 层 55隐 层 40输 出 层 （40种 人）
opts.numepochs = 30;
opts.batchsize = 280;
opts.validation = 1;
opts.plot = 0;
bp2.learningRate = 0.8;
bp2 = nntrain(bp2,input_train,output_train,opts);
%多 分 类 模 型 评 估
an = nnpredict(bp2, input_train);
[erbp2_test ,bad] = nntest(bp2,input_test,output_test);%#ok

figure(2)
plot(output_test,'bo-')
hold on
plot(an,'r*-')
hold on
legend('期望值','预测值')
xlabel('数据组数'),ylabel('值'),title('BP神经网络测试集预测值和期望值的误差对比')