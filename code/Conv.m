function y = Conv(x, W)
%
%

[wrow, wcol, numFilters] = size(W);  %获取每一个过滤器的行数、列数和过滤器的个数
[xrow, xcol, ~         ] = size(x);  %获取每一个被卷积图像矩阵的行数和列数

yrow = xrow - wrow + 1;              %计算最后得到特征矩阵的行数和列数
ycol = xcol - wcol + 1;

y = zeros(yrow, ycol, numFilters);    

for k = 1:numFilters
  filter = W(:, :, k);                    %获取第k个过滤器
  filter = rot90(squeeze(filter), 2);     %去除掉过滤器中维度为1的矩阵，并且旋转180度
  y(:, :, k) = conv2(x, filter, 'valid'); %进行卷积运算
end

end

