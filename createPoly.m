function [R] = createPoly(X, p)
%This function adds higher power features(up to p-th power) of each feature in the data set. 

 [m, n] = size(X); % get the number of rows/cols of X matrix
 C= []; % Initilize an empty matrix
 for i=1:n %Loop through each feature present in X
    x = X(:, i); % pick the ith feature vector
    x_p = polyFeatures(x, p); % covert the ith feature vector into its higher order polynomial form, upto pth degree, using the function polyFeatures()
    C(:,:,i) = x_p; %Save the higher order polynomial features in a 3D array
    [a, b, c] = size(C); %Get the size of C,
    R = reshape(C,[m, (b*c)]); % Reshape the C into a 2D array, with number of rows = number of training examples, number of cols = nums of all features (including higher power ones)
 end

function [X_poly] = polyFeatures(X, pth_power)
%This function maps X (column vector) into the p-th power column vectors
    row_size = size(X,1);
    X_poly = repmat(X,1,pth_power).^repmat((1:pth_power),row_size,1);
end

end