function [error_CV, error_Train, X_train_n, Y_train, X_test_n, Y_test, X_cv_n, Y_cv, theta, J_values_train] = createHyp(X_train, Y_train, X_test, Y_test, X_cv, Y_cv, poly, lambda)

% ====================== createHyp Function details ======================
% 1. This function takes 8 parameters, X_train, Y_train, X_test, Y_test,
% X_cv, Y_cv, poly, lambda and outputs the validation/training error,
% theta, and normalized polynomial features
% 2. Using the createPoly function (see the createPoly.m file for more details), it adds 
% higher power features(up to poly-th power) of each feature in the data set. 
% 3. It then normalize those features.
% 4. Further, it runs Gradient Descent Algorithm on the ormalized dataset
% 5. Using the obtained theta values, it calculates training and validation error

%  Creating pth power features
X_train = createPoly(X_train, poly);
X_test = createPoly(X_test, poly);
X_cv = createPoly(X_cv, poly);

m_train = length(Y_train); % No. of training examples for train set
m_test = length(Y_test); % No. of training examples for test set
m_cv = length(Y_cv); % No. of training examples for CV set

% Feature Normalization
X_train_n = featureNorm(X_train); %Normalize featuers of training set
X_test_n = featureNorm(X_test); %Normalize featuers of testing set
X_cv_n = featureNorm(X_cv); %Normalize featuers of testing set

% Create intercept term to X_train_n and X_test_n
intercept_train = createMatrix(m_train, 1, 1); 
intercept_test = createMatrix(m_test, 1, 1);
intercept_cv = createMatrix(m_cv, 1, 1);

% Add intercept term to X_train_n and X_test_n
X_train_n = [intercept_train X_train_n];
X_test_n = [intercept_test X_test_n];
X_cv_n = [intercept_cv X_cv_n];

%  Running Gradient Descent
% fprintf('Running Gradient Descent ...\n');
alpha = 0.03; % alpha is the learning rate

[theta, J_values_train] = GD(X_train_n, Y_train, alpha, lambda); %Run Gradient Descent 
% fprintf('Gradient Descent DONE.\n');

% Calculating training and validation error
error_CV = calCost(X_cv_n, Y_cv, theta, 0);
error_Train = calCost(X_train_n, Y_train, theta, 0);

end