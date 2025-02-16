clc;
close all;
data = [3, 4, 0; 4, 5, 0; 5, 6, 0; 6, 3, 1; 5, 1, 1; 
        8, 2, 0; 7, 3, 1; 4, 9, 1; 9, 1, 1; 8, 6, 1];

test = [5, 3];
K = 5;
distances = sqrt(sum((data(:, 1:2) - test).^2, 2));
[dist, indices] = sort(distances);
class_counts = zeros(1, 2);

for i = 1:K
    class_label = data(indices(i), 3);
    class_counts(class_label + 1) = class_counts(class_label + 1) + 1;
end
[~, predicted_class] = max(class_counts);
predicted_class = predicted_class - 1;

disp('Class counts for the K nearest neighbors:');
disp(class_counts);
disp('Predicted class:');
disp(predicted_class);

figure;
hold on;
scatter(data(data(:, 3) == 0, 1), data(data(:, 3) == 0, 2), 80, 'o', 'blue', ...
    'filled', 'DisplayName', 'Class 0'); % Class 0 with circle symbol
scatter(data(data(:, 3) == 1, 1), data(data(:, 3) == 1, 2), 80, '^', 'green', ...
    'LineWidth', 1.5, 'DisplayName', 'Class 1'); % Class 1 with star symbol
scatter(test(1), test(2), 100, 's', 'red', 'filled', 'DisplayName', 'Test Point');
radius_k1 = dist(1); % Distance to the nearest neighbor (K = 1)
radius_k3 = dist(3); % Distance to the 3rd nearest neighbor (K = 3)
viscircles(test, radius_k1, 'Color', 'black', 'LineWidth', 1.5, 'LineStyle', '-');
viscircles(test, 2.25, 'Color', 'magenta', 'LineWidth', 1.5, 'LineStyle', '--');
text(test(1) + radius_k1 + 0.1, test(2), 'K = 1', 'FontSize', 10, 'Color', 'black');
text(test(1) + radius_k3 + 0.1, test(2), 'K = 3', 'FontSize', 10, 'Color', 'magenta');

axis equal;
axis([0 10 0 10]);
xticks(0:1:10);
yticks(0:1:10);
xlabel('Feature 1');
ylabel('Feature 2');
title(['K = ', num2str(K), ', Predicted Class = ', num2str(predicted_class)]);
legend('show');
grid on;
hold off;

