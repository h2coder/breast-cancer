% classification.m
% This script will use logistic regression to train a classification model
% for breast cancer data

data = csvread('breast-cancer-wisconsin-trust.data');

data_train = data(1:350, :); % Data used for training.  Remainder will be for testing.
% Feature data to be used for testing the model
data_test = data(351:size(data,1), 2:10);

f_data = data_train(:, 2:10); % Holds the feature training data (not sample ID).
y_data = data_train(:, 11)./2 - 1; % Converts classification code to 0 or 1

% Real classifications to test the model's predictions
y_test = data(351:size(data,1), 11)./2 - 1;

m = size(f_data, 1); % Number of training data points
n = size(f_data, 2) + 1; % Number of features plus 1 for theta_0

theta = zeros(n,1); % Initialize the model parameters to zero
X = [ones(m,1) f_data]; % Adds the constant column of 1s to feature data


% Plot Data
figure(1);
plotData(f_data, y_data);
% Put some labels 
hold on;
% Labels and Legend
xlabel('Feature 1')
ylabel('Feature 2')
% Specified in plot order
legend('Benign', 'Malignant')
hold off;

% Compute and display initial cost and gradient
[cost, grad] = costFunction(theta, X, y_data);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y_data)), theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Apply the model to the rest of the data and see how good it looks.
data_test = [ones(size(data_test,1),1) data_test]; % Add the ones to features
h_test = sigmoid(data_test*theta) >= 0.5;

delta = h_test - y_test;
delta = delta .^ 2;

samples = size(delta,1);
wrong = sum(delta)
Accuracy = 1 - wrong/samples

