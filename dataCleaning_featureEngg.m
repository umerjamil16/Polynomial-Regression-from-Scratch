function X = dataCleaning_featureEngg(X)

% Select the lat long of row which has max Price
[~,x] = max(X(:,1)); % x returns the corresponding row number that contains the max price
ref_lat = X(x,16); %get the lat of row number 'x'
ref_lon = X(x,17); %get the long of row number 'y'

% Create Matrices of size length(X(:, 2)) x 1, containgin the ref_lat and
% ref_lon as follows;
ref_lat_M = createMatrix(length(X(:, 2)), 1, ref_lat); 
ref_long_M = createMatrix(length(X(:, 2)), 1, ref_lon);

% calculate the distance between the two lat, long
dist = haversine(X(:, 16), X(:,17), ref_lat_M, ref_long_M); % This calculates the distance of lat, long of every training example from our refrence lat, long point
X(:,20) = dist; %append the new feature vector to our data (The data previously had 19 columns)

% Next step -> Remove following feature vectors/colomns: (Further explained in the Project Report)
% 11: sqft_above
% 13: Yr_built
% 14: Yr_renovated
% 15: zipcode
% 16: lat
% 17: long

X = X(:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18, 19, 20]); %Only include feature vectors other than mentioned above ones(11,13,14,15,16,17)

%Function to convert degree to radians
function rad = radians(degree) 
    rad = degree .* pi / 180;
end

% Implementation of Haversine formulae, used to calculate distance between
% two set of lat, long coordinates
function dist=haversine(lat1, long1, lat2, long2)
    d_lat = radians(lat2-lat1);
    d_long = radians(long2-long1);
    lat1 = radians(lat1);
    lat2 = radians(lat2);
    a = (sin(d_lat./2)).^2 + cos(lat1) .* cos(lat2) .* (sin(d_long./2)).^2;
    dist = 2 .* asin(sqrt(a)); %I didn't multiply with R(Radius of Sphere) because in any case, I will be normalizing all the feature vectors
end

end