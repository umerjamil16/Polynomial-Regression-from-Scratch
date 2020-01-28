%% ================ Part 1: Initialization ================
% Initialization
% Clear console and close all prevously opened figures
clear; 
close all; 
clc 

fprintf('Program Started...\n');
%% ================ Part 2: LOADING DATA ================
% Load Data
fprintf('Loading data ...\n');
filename = "data3.csv";
data = csvread(filename);
fprintf('DATA LOADING DONE. Press enter to continue.\n');
%pause;

%% ================ Part 3: DATA CLEANING + FEATURE ENGINEERING ================

% DATA CLEANING
fprintf('Data Cleaning/Feature Engineering...\n');
data = dataCleaning_featureEngg(data);
fprintf('Data Cleaning/Feature Engineering DONE. Press enter to continue.\n');
%pause;
%% ================ Part 4: DATA SPLITING ================
fprintf('Data Spliting START...\n');

% Testing/training data spliting
rng('default'); %to ensure constant seed of random gen each time the code runs
[m,n] = size(data) ;% get the size of data matrix

% spliting into 60-40
P = 0.60 ; %Spliting 80-20
rnd = randperm(m)  ; %Take the row number vector and randomize the row number in it
data_train = data(rnd(1:round(P*m)-1),:) ; %get 80% of the data
data_test_cv = data(rnd(round(P*m)+1:end),:) ; %get 20% of the data

%further spliting into Test and CV set
P = 0.50 ; %Spliting 80-20
[m,n] = size(data_test_cv);
rnd = randperm(m)  ; %Take the row number vector and randomize the row number in it
data_test = data_test_cv(rnd(1:round(P*m)),:) ; %get 50% of the data
data_cv = data_test_cv(rnd(round(P*m)+1:end),:) ; %get 50% of the data

X_train = data_train(:, 2:end); % get feature vectors 
Y_train = data_train(:, 1); % get label vector

X_test = data_test(:, 2:end); % get feature vectors 
Y_test = data_test(:, 1); % get label vector

X_cv = data_cv(:, 2:end); % get feature vectors 
Y_cv = data_cv(:, 1); % get label vector

fprintf('Data Spliting DONE. Press enter to continue.\n');
%pause;
%% ================ PART 5: MODEL SELECTION ==============================
modelSelection = 1; % Set it to zero if want to skip Model Selection Step
if modelSelection == 1
    fprintf('MODEL SELECTION PROCESS STARTED.\n');

    lambda = 0;
    poly = 20; %Evaluates models upto 20th degree polynomials
    errorCV = []; % to log validation error
    errorTrain = []; % to log training error

    for i=1:poly % Start with 1 degree polynomial and go upto 20 degree polynomial
        fprintf('Evaluating %i Degree Polynomial Model. Please wait....\n', i);
        % createHyp function(see createHyp.m file)  takes 8 parameters, X_train, Y_train, X_test, Y_test,
        % X_cv, Y_cv, poly, lambda and outputs the validation/training error,
        % theta, and normalized polynomial features
        [errorCV(i), errorTrain(i), ~, ~, ~, ~, ~, ~, ~, ~]= createHyp(X_train, Y_train, X_test, Y_test, X_cv, Y_cv, i, lambda);
    end

    for i=1:length(errorCV) %Pring validation errors obtained for all the models
        fprintf('Model with poly %i - Cross Valid Error %f - Cross Valid Error %f\n', i, errorCV(i), errorTrain(i));
    end
    
    % plot model selection curve
    figure(1);
    plot(1:poly, errorCV, 1:poly, errorTrain);
    title('Model Selection curve')
    legend('Cross Validation Error', 'Training Error')
    xlabel('Degree of Polynomial')
    ylabel('Error')
    savefig("modelSelectionCurve.fig")
    fprintf('Press enter to continue.\n');
    %pause;
    
    % get the index of minimum validation error, where index = degree of
    % polynomial. Basically it picks model with minimum validation error
    [M,index] = min(errorCV);
    fprintf('Model with minimum J_cv is of polynomial degree %i\n', index);
end

% Pick Model with of 7th degree polynomial - As it produces minimum
% validation error amongst other evaluated models (see above)
poly = 7;
fprintf('Model picked is of polynomial degree %i\n', poly);

lambda = 0;
% To get the normalized 7th degree polynomial features and Gradient Descent parameters 
% using createHyp function (see createHyp.m file) to get 7th degree
% polynomial features (normalized), value of parameters (theta), Cost
% function history (J_train), validation and training errors
[errorCV, errorTrain, X_train_n, Y_train, X_test_n, Y_test, X_cv_n, Y_cv, theta, J_train]= createHyp(X_train, Y_train, X_test, Y_test, X_cv, Y_cv, poly, lambda);

% Plot the Gradient Descent Covergence Graph 
fprintf('Ploting Covergence Graph.\n');
% To Plot the convergence graph, Gradient Descent
figure(4);
plot(1:numel(J_train), J_train, '-r', 'LineWidth', 1.5);
title('Convergence graph, Gradient Descent')
legend('Training Error')
xlabel('Number of Iterations');
ylabel('Cost Function Value');
savefig("CostFunctionConvergenceCurve.fig")
%pause;

%% ================ Part 6: Learning Curve, CHECKING BIAS/VARIANCE ================================
m = size(X_train_n, 1); % Get the number of training examples
lambda = 0;
alpha = 0.02; % alpha is the learning rate
[error_train, error_val] = learningCurve(X_train_n, Y_train, X_cv_n, Y_cv, lambda, alpha); 

% Plot the Learning Curve
figure(2);
plot(1:3000:m, error_train, 1:3000:m, error_val);
title('Learning curve for polynomial regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
savefig("learningCurve.fig")

fprintf('Learning Curve Generated. Press enter to continue.\n');
%pause;

%% ================ Part 7: Lambda Adjustment ================ 
alpha = 0.02; % alpha is the learning rate

lambda_vals = [0; 0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1; 3; 10]; % diff vals of lambda to test our model

[error_train, error_val] = validationCurve(X_train_n, Y_train, X_cv_n, Y_cv, alpha, lambda_vals);

% Plot Validation Curve
figure(3);
plot(lambda_vals, error_train, lambda_vals, error_val);
legend('Train', 'Cross Validation');
title('Validation Curve')
xlabel('lambda');
ylabel('Error');
savefig("validationCurve.fig")


%% ================ Part 8: PREDICTION ================
% Pridiction using parameters values obtained using Gradient Descent
% Using Training Set
predictedPrice_train = prediction(theta, X_train_n);
% Using Testing Set
predictedPrice_test = prediction(theta, X_test_n);
% Using Validation set
predictedPrice_cv = prediction(theta, X_cv_n);

%% ================ Part 9: MODEL EVALUATION(/METRICS) ================
% To MSE, RMSE and R-Squared using predected price obtained using parameters computed by Gradient
% Descent
fprintf('MODEL EVALUATION...\n');
[r2_train, RMSE_train, MSE_train] = modelEval(predictedPrice_train, Y_train);
[r2_test, RMSE_test, MSE_test] = modelEval(predictedPrice_test, Y_test);
[r2_cv, RMSE_cv, MSE_cv] = modelEval(predictedPrice_cv, Y_cv);
fprintf('MODEL EVAL DONE. Press enter to continue.\n');
%pause;  

fprintf('\n');
fprintf('Evualation Metrics for parameters computed using Gradient Descent:\n');

fprintf('--------------------------------------------------\n');
fprintf('   Dataset  |         MSE             |         RMSE        |    R-Squared         \n');
fprintf('--------------------------------------------------\n');
fprintf('   Training |    %f    |    %f    |     %f \n', MSE_train, RMSE_train, r2_train)
fprintf('   Testing  |    %f    |    %f    |     %f \n', MSE_test, RMSE_test, r2_test)
fprintf('   Validat  |    %f    |    %f    |     %f \n', MSE_cv, RMSE_cv, r2_cv)
fprintf('--------------------------------------------------\n');

fprintf('\n');
fprintf('--- Program Ended ---\n');
