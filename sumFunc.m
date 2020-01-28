function [rslt] = sumFunc(inputM)

rslt = [];
elemSum = 0;
[row_M, col_M] = size(inputM);

%Loop to calculate sum of columns of input matrices
for j=1:col_M
  for i=1:row_M
    elemSum = elemSum + inputM(i,j);
  end
  rslt(:, j) = elemSum;
  elemSum = 0;
end

end