function [error_train, error_val] = learningCurve(X_train, Y_train, X_cv, Y_cv, lambda, alpha)
% learningCurve function (see learningCurve.m file) calculates the
% training/validation errors to generate learning curve graphs. It starts with first ith-training examples, finds the
% corresponding training and validation error, and then increment/increase
% the number of training examples and log the corresponding errors. The
% process is repeated until whole training dataset is covered.

% Number of training examples
m = size(X_train, 1);

error_train = []; % to log training error
error_val   = []; % to log validation error

for i = 1:3000:m % increment the number of training examples by 3000
    fprintf('Calculating thetas for first %i training examples. Please wait... \n', i)
    [theta, ~] = GD(X_train(1:i,:), Y_train(1:i),  alpha, lambda); %Run Gradient Descent 
    error_train(end+1, : ) = calCost(X_train(1:i,:), Y_train(1:i), theta, lambda); %calculates training error on first ith examples
    error_val(end+1, : ) = calCost(X_cv, Y_cv, theta, lambda); %calculates validation error
end


end
