clc;
clear;
close all;

numSamples = 80;
data = randn(2, numSamples) * 10;
means = mean(data, 2);
data_centered = data - means;
n = size(data, 2);
covariance = (data_centered * data_centered') / (n - 1);
[eigen_vectors, eigen_values] = eig(covariance);
[eigen_values_sorted, idx] = sort(diag(eigen_values), 'descend');
eigen_vectors_sorted = eigen_vectors(:, idx);
pca_projection = eigen_vectors_sorted' * data_centered;

disp(covariance);
disp(eigen_values_sorted);
disp(eigen_vectors_sorted);
disp(pca_projection);

figure;
scatter(data(1, :), data(2, :), 50, 'bo', 'filled');
hold on;
scaleFactor = 1;  
quiver(means(1), means(2), eigen_vectors_sorted(1,1) * sqrt(eigen_values_sorted(1)) * scaleFactor, ...
       eigen_vectors_sorted(2,1) * sqrt(eigen_values_sorted(1)) * scaleFactor, 0, 'r', 'LineWidth', 2);
quiver(means(1), means(2), eigen_vectors_sorted(1,2) * sqrt(eigen_values_sorted(2)) * scaleFactor, ...
       eigen_vectors_sorted(2,2) * sqrt(eigen_values_sorted(2)) * scaleFactor, 0, 'g', 'LineWidth', 2);
xlabel('X-axis');
ylabel('Y-axis');
title('Original Data and Principal Components');
legend('Original Data', 'PC1', 'PC2');
grid on;
hold off;

principal_component_1 = eigen_vectors_sorted(:,1);
projected_1D = principal_component_1' * data_centered;
reconstructed_2D = means + principal_component_1 * projected_1D;

disp(projected_1D);

figure;
scatter(reconstructed_2D(1, :), reconstructed_2D(2, :), 50, 'rx', 'LineWidth', 2);
hold on;
plot([means(1) - 2 * eigen_vectors_sorted(1,1), means(1) + 2 * eigen_vectors_sorted(1,1)], ...
     [means(2) - 2 * eigen_vectors_sorted(2,1), means(2) + 2 * eigen_vectors_sorted(2,1)], 'k--', 'LineWidth', 1.5);
xlabel('X-axis');
ylabel('Y-axis');
title('Data Projected onto Principal Eigenvector');
legend('Projected Data Points', 'Principal Eigenvector');
grid on;
hold off;
