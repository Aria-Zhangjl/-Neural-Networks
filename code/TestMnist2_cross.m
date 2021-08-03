%һ����������ز�
%�н�����֤

%����ѵ�����������ǩ
Images1 = loadMNISTImages('D:\��װ���ݼ�\train-images-idx3-ubyte');
Images1 = reshape(Images1, 28, 28, []);
Labels1 = loadMNISTLabels('D:\��װ���ݼ�\train-labels-idx1-ubyte');
Labels1(Labels1 == 0) = 10;    % 0 --> 10
X1 = Images1(:, :, 1:60000);
D1 = Labels1(1:60000);
%������Լ��������ǩ
Images2 = loadMNISTImages('D:\��װ���ݼ�\t10k-images-idx3-ubyte');
Images2 = reshape(Images2, 28, 28, []);
Labels2 = loadMNISTLabels('D:\��װ���ݼ�\t10k-labels-idx1-ubyte');
Labels2(Labels2 == 0) = 10;    % 0 --> 10
X2 = Images2(:, :, 1:10000);
D2 = Labels2(1:10000);
%��������
X = cat(3,X1,X2);
D = [D1;D2];

%��ȡ��ʼ��Ȩ�ؾ���
load('Cross_weight2.mat')

Accuracy=0;
%ʮ�۽�����֤
indices = crossvalind('Kfold', 70000, 10);
for k=1:10
    test = (indices == k); 
    train = ~test;
	
	%ȡ��ѵ������
    train_data=X(:,:,train);
    train_target=D(train,:);
	
    %ÿһ����֤����3�ֵ�ѵ��
    for epoch = 1:3
        epoch
        [W1, W5,Wo] = MnistConv_Textbook(W1,W5,Wo, train_data, train_target);
    end
	
    %ȡ����������
    test_data=X(:,:,test);
    test_target=D(test,:);
	
	%��ʼ����
    acc = 0;
    N   = length(test_target);
    for j = 1:N
     x  = test_data(:,:, j);        % 28x28������
     y1 = Conv(x, W1);              % ����㣬���20x20x20�ľ���
     y2 = ReLU(y1);                 %
     y3 = Pool(y2);                 % �ػ��㣬���10x10x20
     y4 = reshape(y3, [], 1);       %
     v5 = W5*y4;                    % 
     y5 = ReLU(v5);                 % ���ز㣬���100x1������
     v  = Wo*y5;                    % ����㣬���10x1������
     y  = Softmax(v);        

     [~, i] = max(y);               % ������֤
     if i == test_target(j)
        acc = acc + 1;
     end
    end
    acc = acc / N;
    Accuracy=Accuracy+acc;
    fprintf( '%d :Accuracy is %f\n',k,acc);  %���ÿ����֤��׼ȷ��
end

%����ƽ��׼ȷ�ʣ�������ģ��
Accuracy = Accuracy/10;
fprintf('Average accuracy is %f\n', Accuracy);
save('show2.mat');