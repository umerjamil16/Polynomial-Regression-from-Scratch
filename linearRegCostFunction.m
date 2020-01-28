function [J] = linearRegCostFunction(X, y, theta)

h = X*theta;

J = (sum((h-y).^2));

end
