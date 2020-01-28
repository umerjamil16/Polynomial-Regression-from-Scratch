%Code for matrix multiplication
% Clear screen, clear previous variables and closes all figures
function [X] = createMatrix(row, col, elemVal)

X = [];

for j=1:col
    for i=1:row
        X(i, j)= elemVal;
    end
end

end