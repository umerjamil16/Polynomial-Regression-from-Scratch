%Function to calculate Mean
function meanVal = meanFunc(InputM)
%to calculate mean across columns of Input Matrix, Returns a row vector
%containing Mean of columns of Input Matrix
    m = length(InputM);
    sumVal = sumFunc(InputM);
    meanVal = sumVal/m;
end
