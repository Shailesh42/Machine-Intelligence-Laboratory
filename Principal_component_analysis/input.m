clc;
clear;
close all;
data = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1; 
        2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9];
means = mean(data, 2);
data_centered = data - means;
n = size(data, 2);
covariance = (data_centered * data_centered') / (n - 1);
[eigen_vectors, eigen_values] = eig(covariance);
[eigen_values_sorted, idx] = sort(diag(eigen_values), 'descend');
eigen_vectors_sorted = eigen_vectors(:, idx);
pca_projection = eigen_vectors_sorted' * data_centered;
disp('Covariance Matrix:');
disp(covariance);
disp('Eigenvalues of the Covariance Matrix:');
disp(eigen_values_sorted);
disp('Eigenvectors (Principal Components):');
disp(eigen_vectors_sorted);
disp('Projected Data onto Principal Components:');
disp(pca_projection);
figure;
scatter(data(1, :), data(2, :), 50, 'bo', 'filled');
hold on;

% Plot principal component vectors
quiver(means(1), means(2), eigen_vectors_sorted(1,1) * eigen_values_sorted(1), ...
       eigen_vectors_sorted(2,1) * eigen_values_sorted(1), 2, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(means(1), means(2), eigen_vectors_sorted(1,2) * eigen_values_sorted(2), ...
       eigen_vectors_sorted(2,2) * eigen_values_sorted(2), 2, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);

% Annotate the main eigenvalue on the principal eigenvector
text(means(1) + eigen_vectors_sorted(1,1) * eigen_values_sorted(1), ...
     means(2) + eigen_vectors_sorted(2,1) * eigen_values_sorted(1), ...
     sprintf('λ1=%.2f', eigen_values_sorted(1)), 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');

xlabel('X-axis');
ylabel('Y-axis');
title('Original Data and Principal Components');
legend('Original Data', 'PC1 (Principal Eigenvector)', 'PC2');
grid on;
hold off;

% Step 8: Project all data onto the main principal eigenvector (PC1)
principal_component_1 = eigen_vectors_sorted(:,1);
projected_1D = principal_component_1' * data_centered;

% Convert 1D projections back to 2D space along the principal eigenvector
reconstructed_2D = means + principal_component_1 * projected_1D;

% Display the 2D projected values on the main principal eigenvector
disp('Projected values along Principal Eigenvector (1D scalar values):');
disp(projected_1D);

% Step 9: Plot the 2D projection along the main eigenvector
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
