function [error_train, error_val] = validationCurve(X_train, Y_train, X_cv, Y_cv, alpha, lambda_values)

% Initilize corresponding training/validaiton matrices for log.
error_train = zeros(length(lambda_values), 1);
error_val = zeros(length(lambda_values), 1);

for i = 1:length(lambda_values) % loop through all the passed lambda values
    [theta, ~] = GD(X_train, Y_train, alpha, lambda_values(i)); % Find theta vals using Gradient Descent
    error_train(i) = calCost(X_train, Y_train, theta, 0); % find training error
    error_val(i) = calCost(X_cv, Y_cv, theta, 0); % find training error
end

end
