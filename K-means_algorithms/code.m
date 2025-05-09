clc;
clear;
close all;

% Dataset
dataset = [1,1 ; 1.5,2 ; 3,4 ; 5,7 ; 3.5,5 ; 4.5,5 ; 3.5,4.5];

% Number of clusters
k = 2;

% Initial centroids
centroids = [dataset(1,:); dataset(4,:)];

% Number of data points
n = size(dataset, 1);
cluster_assignments = zeros(n, 1);
old_centroids = zeros(size(centroids));

while ~isequal(centroids, old_centroids)
    old_centroids = centroids; 
    distance_table = zeros(n, k);

    % Calculate distances and assign clusters
    for i = 1:n
        for j = 1:k
            distance_table(i, j) = sqrt(sum((dataset(i,:) - centroids(j,:)).^2)); 
        end
        [~, cluster_assignments(i)] = min(distance_table(i, :)); 
    end

    % Update centroids
    for j = 1:k
        points_in_cluster = dataset(cluster_assignments == j, :);
        if ~isempty(points_in_cluster)
            centroids(j, :) = mean(points_in_cluster, 1); 
        end
    end
end

% Display the results
fprintf('Final Centroids:\n');
disp(array2table(centroids, 'VariableNames', {'X', 'Y'}));

fprintf('Cluster Assignments:\n');
disp(array2table([(1:n)', cluster_assignments], 'VariableNames', {'DataPoint', 'Cluster'}));

fprintf('Distance Table:\n');
distance_table = array2table(distance_table, 'VariableNames', {'Centroid_1', 'Centroid_2'});
disp(distance_table);

% Plotting the clusters
figure;
hold on;
colors = lines(k);
for j = 1:k
    points_in_cluster = dataset(cluster_assignments == j, :);
    scatter(points_in_cluster(:,1), points_in_cluster(:,2), 50, colors(j,:), 'filled');
end

% Plot centroids
scatter(centroids(:,1), centroids(:,2), 100, 'k', 'x', 'LineWidth', 2);
for j = 1:k
    points_in_cluster = dataset(cluster_assignments == j, :);
    if size(points_in_cluster, 1) > 1
        cluster_radius = max(sqrt(sum((points_in_cluster - centroids(j,:)).^2, 2))); 
    else
        cluster_radius = 0.5;  % Assign a default small radius if only one point exists
    end
    viscircles(centroids(j,:), cluster_radius + 0.5, 'LineStyle', '-', 'EdgeColor', colors(j,:));
end
xlabel('X-axis');
ylabel('Y-axis');
title('K-means Clustering Visualization');
legend_strings = arrayfun(@(x) sprintf('Cluster %d', x), 1:k, 'UniformOutput', false);
legend([legend_strings, {'Centroids'}], 'Location', 'best');
hold off;

