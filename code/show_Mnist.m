%�㱨ʱչ������������Ĳ��Խ��



%�ֱ��������������Ĳ���

%��װ�����
label={'ţ�п�','����',',ȹ��','����','��Ь','����','�˶�Ь','��','��ѥ','T��'};

%��һ�����������ز�

%��������ͼƬ
fig_file = open('test_figure.fig');
m = fig_file .CurrentAxes.Children.CData;

% ��ȡ��һ���������ģ�Ͳ���
load('show1.mat');               
m=reshape(m,784,1);

%��ʼ����
v1=W1*m;
y1=ReLU(v1);
v2=W2*y1;
y2=ReLU(v2);
v3=W3*y2;
y3=ReLU(v3);
v4=W4*y3;
y4=Softmax(v4);
[~, i] = max(y4);
%���������
fprintf('1:');
disp(label(i));


%�ڶ�����һ��������һ�����ز�

%��������ͼƬ
m = fig_file .CurrentAxes.Children.CData;

% ��ȡ�ڶ����������ģ�Ͳ���
load('show2.mat'); 

%��ʼ����
y1 = Conv(m, W1);              
y2 = ReLU(y1);                 
y3 = Pool(y2);                 
y4 = reshape(y3, [], 1);       
v5 = W5*y4;                   
y5 = ReLU(v5);                 
v  = Wo*y5;                    
y  = Softmax(v);        
[~, i] = max(y);

%���������
fprintf('2:');
disp(label(i));

%�������������������������ز�
m = fig_file .CurrentAxes.Children.CData;

% ��ȡ�������������ģ�Ͳ���
load('show3.mat');               


%��ʼ����
y1 = Conv(m, W1);              % Convolution1,  26x26x1
y2 = ReLU(y1);                 %
y3 = Conv(y2,W2);               % Convolution2,  16x16x32
y4 = ReLU(y3);
y5 = Pool(y4);                 % Pooling,      8x8x32
y6 = reshape(y5, [], 1);       %ת��2048*1����
v3 = W3*y6;                    % ReLU,             2000
y7 = ReLU(v3);                 %
v4 = W4*y7;
y8 = ReLU(v4);
v  = W5*y8;                    % Softmax,          10x1
y  = Softmax(v);   
%���������
fprintf('3:');
disp(label(i));