function [r2, RMSE, MSE] = modelEval(Predicted, Actual)

MSE = (meanFunc(( Actual - Predicted).^2))/2;  % Root Mean Squared Error

RMSE = sqrt(MSE);  % Root Mean Squared Error

%Implementation of formulae of R-Squared
a=sumFunc((Actual-Predicted).^2);
b=sumFunc((Actual-meanFunc(Actual)).^2);    
r2=1 - a/b; 

end