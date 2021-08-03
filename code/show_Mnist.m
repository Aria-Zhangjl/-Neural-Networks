%汇报时展现三个神经网络的测试结果



%分别进行三个神经网络的测试

%服装的类别
label={'牛仔裤','套衫',',裙子','外套','凉鞋','衬衫','运动鞋','包','短靴','T恤'};

%第一个：三层隐藏层

%读出测试图片
fig_file = open('test_figure.fig');
m = fig_file .CurrentAxes.Children.CData;

% 获取第一个神经网络的模型参数
load('show1.mat');               
m=reshape(m,784,1);

%开始测试
v1=W1*m;
y1=ReLU(v1);
v2=W2*y1;
y2=ReLU(v2);
v3=W3*y2;
y3=ReLU(v3);
v4=W4*y3;
y4=Softmax(v4);
[~, i] = max(y4);
%输出分类结果
fprintf('1:');
disp(label(i));


%第二个：一个卷积层和一个隐藏层

%读出测试图片
m = fig_file .CurrentAxes.Children.CData;

% 获取第二个神经网络的模型参数
load('show2.mat'); 

%开始测试
y1 = Conv(m, W1);              
y2 = ReLU(y1);                 
y3 = Pool(y2);                 
y4 = reshape(y3, [], 1);       
v5 = W5*y4;                   
y5 = ReLU(v5);                 
v  = Wo*y5;                    
y  = Softmax(v);        
[~, i] = max(y);

%输出分类结果
fprintf('2:');
disp(label(i));

%第三个：两个卷积层和两个隐藏层
m = fig_file .CurrentAxes.Children.CData;

% 获取第三个神经网络的模型参数
load('show3.mat');               


%开始测试
y1 = Conv(m, W1);              % Convolution1,  26x26x1
y2 = ReLU(y1);                 %
y3 = Conv(y2,W2);               % Convolution2,  16x16x32
y4 = ReLU(y3);
y5 = Pool(y4);                 % Pooling,      8x8x32
y6 = reshape(y5, [], 1);       %转成2048*1矩阵
v3 = W3*y6;                    % ReLU,             2000
y7 = ReLU(v3);                 %
v4 = W4*y7;
y8 = ReLU(v4);
v  = W5*y8;                    % Softmax,          10x1
y  = Softmax(v);   
%输出分类结果
fprintf('3:');
disp(label(i));