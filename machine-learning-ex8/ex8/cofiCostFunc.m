function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


totalCosts = (X*Theta'-Y).^2;
totalCosts = R.*(totalCosts);
J = sum(sum(totalCosts))/2;

[m n] = size(X);
for i = 1:m
	idx = find(R(i,:)==1);
	ThetaTemp = Theta(idx,:);
	YTemp = Y(i,idx);
	errorFac = (X(i,:)*ThetaTemp' - YTemp);
	X_grad(i,:) = errorFac*ThetaTemp;
	regXk = X(i,:).*lambda;
	X_grad(i,:) = X_grad(i,:) + regXk;
endfor

[p q] = size(Theta);
for i = 1:p
	idx = find(R(:,i)==1);
	ThetaTemp = Theta(i,:);
	XTemp = X(idx,:);
	YTemp = Y(idx,i);
	errorFac = (XTemp*Theta(i,:)' - YTemp);
	Theta_grad(i,:) = XTemp'*errorFac;
	regThetak = Theta(i,:).*lambda;
	Theta_grad(i,:) = Theta_grad(i,:) + regThetak;
	

endfor


thetaReg = sum(sum(Theta.^(2)))*(lambda/2); 
xReg = sum(sum(X.^(2)))*(lambda/2);
J= J + thetaReg + xReg;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
