function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

param_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
results = zeros(length(param_vec)^2,3);
k=1;
for i=1:length(param_vec)
	for j=1:length(param_vec)
		C = param_vec(i);
		sigma = param_vec(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		prediction = svmPredict(model,Xval);
		err = mean(double(prediction ~= yval));
		results(k,:) = [C,sigma,err];
		k=k+1;
	endfor
endfor
[x,ix] = min(results(:,3));
best = results(ix,:);


C = best(1);
sigma = best(2);
		




% =========================================================================

end
