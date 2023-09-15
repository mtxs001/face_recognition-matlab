clc;clear;close all;
load('ORL_32x32.mat');
% fea=uint8(fea);
% imshow(fea);
% %% A为便于观察的图片
% A=zeros(1280,320);
% for n=0:39
%     for i=0:31    
%         for j=1:10
%             A(32*n+i+1,(j-1)*32+1:j*32)=fea(10*n+j,i*32+1:(i+1)*32);
%         end
%     end
% end
% A=uint8(A);
% imshow(A);
%% gnd1为戴眼镜标识
gnd1=zeros(400,1);
data1=[11:20,31,32,33,38,39,51:60,64,65,70,121,122,125:140,161,162,165:170,181,182,183,186,190:193,195,197:200,261:280,301:310,331:340,359,360:370];
for i=data1  
    gnd1(i,1)=1;
end
%% 训练集测试集
p=0.3;
[Train,Test]=crossvalind('Holdout',gnd1,p);
input_train=fea(Train,:);
output_train=gnd1(Train,:);
input_test=fea(Test,:);
output_test=gnd1(Test,:);
%% 构建BP神经网络
cengshu=[5 20 5];
net=newff(input_train',output_train',cengshu,{'logsig','logsig'},'trainlm');%gd
net.trainParam.epochs=100;        %训练次数
net.trainParam.lr=0.2;            %学习速率
net.trainParam.goal=0.000000000001;      %训练目标最小误差


%% BP神经网络训练
net=train(net,input_train',output_train');
% 保存模型
save bestnet net;
load('bestnet');
%% BP神经网络测试
an=sim(net,input_test');            %用训练好的模型进行仿真
ans1=round(an);                               

figure(1)
plot(output_test,'bo-')
hold on
plot(an,'r*-')
hold on
legend('期望值','预测值')
xlabel('数据组数'),ylabel('值'),title('BP神经网络测试集预测值和期望值的误差对比')
    
%% 计算混淆矩阵、测试精度，查准率，查全率和F1值
ans1=ans1';
cm=[sum(output_test==1 & ans1==1),sum(output_test==1 & ans1==0);sum(output_test==0 & ans1==1),sum(output_test==0 & ans1==0)];
ex=(cm(1,1)+cm(2,2))/sum(sum(cm));
P=cm(1,1)/(cm(1,1)+cm(2,1));
R=cm(1,1)/(cm(1,1)+cm(1,2));
F1=2*P*R/(P+R);
cm=[sum(output_test==1 & ans1==1),sum(output_test==1 & ans1==0),sum(output_test==0 & ans1==1),sum(output_test==0 & ans1==0)];
cm=int2str(cm);
cengshu=int2str(cengshu);

data11 = readcell('data_1.xlsx');
data12 ={cengshu,cm,ex,P,R,F1};
xlswrite('data_1.xlsx',data12,'sheet1','A1')
xlswrite('data_1.xlsx',data11,'sheet1','A2')
