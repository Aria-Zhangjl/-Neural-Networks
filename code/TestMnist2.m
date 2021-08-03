%一个卷积和隐藏层
%没有交叉验证

%导入训练集数据和标签
Images1 = loadMNISTImages('D:\服装数据集\train-images-idx3-ubyte');
Images1 = reshape(Images1, 28, 28, []);
Labels1 = loadMNISTLabels('D:\服装数据集\train-labels-idx1-ubyte');
Labels1(Labels1 == 0) = 10;    % 0 --> 10
X = Images1(:, :, 1:60000);
D = Labels1(1:60000);

%初始化权重矩阵
rng(1);
W1 = 1e-2*randn([9 9 20]);
rng(1);
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
rng(1);
Wo = (2*rand( 10,  100) - 1) * sqrt(6) / sqrt( 10 +  100);

%进行三轮的训练
for epoch = 1:3
    epoch
    [W1, W5,Wo] = Train_Mnist2(W1,W5,Wo, X,D);
end

%保存训练好的模型
save('Cross_weight2.mat');

%导入测试集数据和标签
Images2 = loadMNISTImages('D:\服装数据集\t10k-images-idx3-ubyte');
Images2 = reshape(Images2, 28, 28, []);
Labels2 = loadMNISTLabels('D:\服装数据集\t10k-labels-idx1-ubyte');
Labels2(Labels2 == 0) = 10;    % 0 --> 10
X = Images2(:, :, 1:10000);
D = Labels2(1:10000);

%进行测试
acc = 0;
N   = length(D);
for j = 1:N
  x  = X(:,:, j);                % 输入层，得到28x28的矩阵
  y1 = Conv(x, W1);              % 卷积层，得到20x20x20的矩阵
  y2 = ReLU(y1);                 %
  y3 = Pool(y2);                 % 池化层，输出10x10x20
  y4 = reshape(y3, [], 1);       %
  v5 = W5*y4;                    % 
  y5 = ReLU(v5);                 % 隐藏层，输出100x1的向量
  v  = Wo*y5;                    % 输出层，输出10x1的向量
  y  = Softmax(v);        
  [~, i] = max(y);               %进行验证
  if i == D(j)
     acc = acc + 1;
  end
end
acc = acc / N;
fprintf( 'Accuracy is %f\n',acc);  %输出正确率
