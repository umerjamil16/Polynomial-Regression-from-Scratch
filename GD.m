% Implementation of  GRADIENT DESCENT
function [theta, J_values_train, J_values_cv] = GD(X_train, Y_train, alpha, lambda)
%mulFunc(A, B) multiple two matrices A and B and returns the resulting Matrice

[~, q] = size(X_train); %get number of columns/features of training set 
theta = createMatrix(q, 1, 0);%zeros(q, 1);

m = length(Y_train); % number of training examples
J_values_train = []; % To log values of Cost Function J

threshold = 0.01;
iter = 0;
while 1
    grad = mulFunc((X_train'), (mulFunc(X_train, theta) - Y_train)) + lambda*theta;
    grad(1) = grad(1) - lambda*theta(1);
    theta = theta - (alpha/m)*(grad);
    iter = iter + 1;
    J_values_train(iter ) = calCost(X_train, Y_train, theta, lambda); %To compute cost function

     fprintf('Current iter: %i and CostFuncVal: %f\n', iter, J_values_train(iter)) 
    if isinf(J_values_train(iter))
        fprintf('Cost Function Value INCREASING. Please re-adjust learning rate parameter...\n'); 
        fprintf('Current Learning Rate is %f...\n', alpha);
        pause;
    end

    if iter > 5
        if (abs(((J_values_train((end-1)) - J_values_train(end))*100)/J_values_train(end-1)) < threshold)
            break %break while loop if percentage increase in J_values is less than threshold (0.01)
        end
    end
end

end