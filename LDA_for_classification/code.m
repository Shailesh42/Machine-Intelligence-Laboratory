clc; clear; close all;
data = [2.95, 2.53, 3.57, 3.16, 2.58, 2.16, 3.27; 
        6.63, 7.79, 5.65, 5.47, 4.46, 6.22, 3.52];
output = [1, 1, 1, 1, 0, 0, 0]; 
test_point = [2.81; 5.46];       
class1 = data(:, output == 1);
class0 = data(:, output == 0);
overall_mean = mean(data, 2);
mean1 = mean(class1, 2);
mean0 = mean(class0, 2);
cov_class1 = (class1 - mean1) * (class1 - mean1)' / size(class1, 2);
cov_class0 = (class0 - mean0) * (class0 - mean0)' / size(class0, 2);
prior1 = sum(output == 1) / length(output);
prior0 = sum(output == 0) / length(output);

cov_matrix = prior1 * cov_class1 + prior0 * cov_class0;
inv_cov_matrix = inv(cov_matrix);
g1 = mean1' * inv_cov_matrix * test_point - 0.5 * mean1' * inv_cov_matrix * mean1 + log(prior1);
g0 = mean0' * inv_cov_matrix * test_point - 0.5 * mean0' * inv_cov_matrix * mean0 + log(prior0);

if g1 > g0
    class_assigned = 1;
    decision = 'Passed';
else
    class_assigned = 0;
    decision = 'Not Passed';
end
fprintf("Discriminant function for Class 1 (Passed): %.4f\n", g1);
fprintf("Discriminant function for Class 0 (Not Passed): %.4f\n", g0);
fprintf("Test point (%.2f, %.2f) assigned to Class: %d (%s)\n", test_point(1), test_point(2), class_assigned, decision);
figure;
hold on; grid on;
scatter(class1(1, :), class1(2, :), 'ro', 'filled');
scatter(class0(1, :), class0(2, :), 'bs', 'filled'); 
scatter(test_point(1), test_point(2), 'kp', 'filled', 'SizeData', 100); 

x_vals = linspace(min(data(1, :)) - 0.5, max(data(1, :)) + 0.5, 100);
y_vals = linspace(min(data(2, :)) - 0.5, max(data(2, :)) + 0.5, 100);
[X, Y] = meshgrid(x_vals, y_vals);
Z = zeros(size(X));

for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        point = [X(i, j); Y(i, j)];
        g1_temp = mean1' * inv_cov_matrix * point - 0.5 * mean1' * inv_cov_matrix * mean1 + log(prior1);
        g0_temp = mean0' * inv_cov_matrix * point - 0.5 * mean0' * inv_cov_matrix * mean0 + log(prior0);
        Z(i, j) = g1_temp - g0_temp;
    end
end

contour(X, Y, Z, [0, 0], 'k', 'LineWidth', 2); 

legend('Class 1 (Passed)', 'Class 0 (Failed)', 'Test Point', 'Decision Boundary', 'Location', 'best');
title('Linear Discriminant Analysis with Decision Boundary');
xlabel('Feature 1');
ylabel('Feature 2');
hold off;
disp('Overall Mean of Data:');
disp(overall_mean);
disp('Mean of Class 1 (Passed):');
disp(mean1);
disp('Mean of Class 0 (Not Passed):');
disp(mean0);
disp('Covariance matrix of Class 1 (Passed):');
disp(cov_class1);
disp('Covariance matrix of Class 0 (Not Passed):');
disp(cov_class0);
disp('Common Covariance Matrix:');
disp(cov_matrix);
disp('Prior Probability of Class 1 (Passed):');
disp(prior1);
disp('Prior Probability of Class 0 (Not Passed):');
disp(prior0);
disp('Final Classification Result:');
fprintf("Test point assigned to class %d (%s) with g1 = %.4f and g0 = %.4f\n", class_assigned, decision, g1, g0);
