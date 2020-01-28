function [rsltMatrix] = mulFunc(A, B)

% The size of each matrix is considered for these calculations
[row_A col_A] = size(A);
[row_B col_B] = size(B);

rsltMatrix = []; %Contains the resulting matrix

if row_B ~= col_A
    fprintf('Input Matrices not compatiable for Multiplication');
    pause;
end

% Implementation of Matrixes Multiplication code
for p = 1 : row_A
    for q = 1 : col_B
        elemSum = 0;
        for r = 1 : col_A
            elemSum = elemSum + A(p,r) * B(r,q);
        end
        rsltMatrix(p,q) = elemSum;
    end
end

end