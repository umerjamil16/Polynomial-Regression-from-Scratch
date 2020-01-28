function feature_norm = featureNorm(inputFeature)

[~,q] = size(inputFeature); % get numbre of cols of inputFeature
mu =  createMatrix(1, q, 0);%zeros(1, q); %To contain mean of each feature
sigma = createMatrix(1, q, 0);%zeros(1, q); %To contain Standard Deviation of each feature

mu = meanFunc(inputFeature); %calculate mean of inputFeature Vector
sigma = stdCal(inputFeature); %calculate stadard deviation of inputFeature vector
feature_norm = (inputFeature - mu)./sigma; %Feature standardization 

% Implementation of Standard Deviation Formulae Calculation
function SD = stdCal(input) 
       SD = sqrt(sumFunc((input-meanFunc(input)).^2/(length(input)-1))); 
end

end
