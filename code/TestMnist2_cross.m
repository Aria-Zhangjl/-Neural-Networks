%一个卷积和隐藏层
%有交叉验证

%导入训练集数据与标签
Images1 = loadMNISTImages('D:\服装数据集\train-images-idx3-ubyte');
Images1 = reshape(Images1, 28, 28, []);
Labels1 = loadMNISTLabels('D:\服装数据集\train-labels-idx1-ubyte');
Labels1(Labels1 == 0) = 10;    % 0 --> 10
X1 = Images1(:, :, 1:60000);
D1 = Labels1(1:60000);
%导入测试集数据与标签
Images2 = loadMNISTImages('D:\服装数据集\t10k-images-idx3-ubyte');
Images2 = reshape(Images2, 28, 28, []);
Labels2 = loadMNISTLabels('D:\服装数据集\t10k-labels-idx1-ubyte');
Labels2(Labels2 == 0) = 10;    % 0 --> 10
X2 = Images2(:, :, 1:10000);
D2 = Labels2(1:10000);
%数据整合
X = cat(3,X1,X2);
D = [D1;D2];

%获取初始化权重矩阵
load('Cross_weight2.mat')

Accuracy=0;
%十折交叉验证
indices = crossvalind('Kfold', 70000, 10);
for k=1:10
    test = (indices == k); 
    train = ~test;
	
	%取出训练数据
    train_data=X(:,:,train);
    train_target=D(train,:);
	
    %每一次验证进行3轮的训练
    for epoch = 1:3
        epoch
        [W1, W5,Wo] = MnistConv_Textbook(W1,W5,Wo, train_data, train_target);
    end
	
    %取出测试数据
    test_data=X(:,:,test);
    test_target=D(test,:);
	
	%开始测试
    acc = 0;
    N   = length(test_target);
    for j = 1:N
     x  = test_data(:,:, j);        % 28x28的输入
     y1 = Conv(x, W1);              % 卷积层，输出20x20x20的矩阵
     y2 = ReLU(y1);                 %
     y3 = Pool(y2);                 % 池化层，输出10x10x20
     y4 = reshape(y3, [], 1);       %
     v5 = W5*y4;                    % 
     y5 = ReLU(v5);                 % 隐藏层，输出100x1的向量
     v  = Wo*y5;                    % 输出层，输出10x1的向量
     y  = Softmax(v);        

     [~, i] = max(y);               % 进行验证
     if i == test_target(j)
        acc = acc + 1;
     end
    end
    acc = acc / N;
    Accuracy=Accuracy+acc;
    fprintf( '%d :Accuracy is %f\n',k,acc);  %输出每轮验证的准确率
end

%计算平均准确率，并保存模型
Accuracy = Accuracy/10;
fprintf('Average accuracy is %f\n', Accuracy);
save('show2.mat');