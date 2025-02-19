clc;
clear;
close all;

numSamples = 80;
group1 = randn(1, numSamples) * 2.5 + 3;
group2 = randn(1, numSamples) * 1.5 + 7;
dataPoints = [group1, group2];
numData = length(dataPoints);
figure();
hold on ;
scatter(dataPoints(1, :), 30, 'ro', 'filled');
hold off;
means = [4, 6];
vars = [0.4, 0.5];
mixWeightA = 0.5;
mixWeightB = 0.5;
threshold = 1e-4;

while true
    probA = (1 / sqrt(2 * pi * vars(1))) * exp(-((dataPoints - means(1)).^2) / (2 * vars(1)));
    probB = (1 / sqrt(2 * pi * vars(2))) * exp(-((dataPoints - means(2)).^2) / (2 * vars(2)));
    respA = (mixWeightA * probA) ./ (mixWeightA * probA + mixWeightB * probB);
    respB = 1 - respA; 
    sumRespA = sum(respA);
    sumRespB = sum(respB);
    
    new_means = [sum(respA .* dataPoints) / sumRespA, sum(respB .* dataPoints) / sumRespB];
    new_vars = [sum(respA .* (dataPoints - means(1)).^2) / sumRespA, sum(respB .* (dataPoints - means(2)).^2) / sumRespB];
    
    mixWeightA = sumRespA / numData;
    mixWeightB = sumRespB / numData;
    
    if all(abs(means - new_means) < threshold) && all(abs(vars - new_vars) < threshold)
        break;
    end
    
    means = new_means;
    vars = new_vars;
end

finalClusters = respA > 0.5;

figure; hold on; grid on;
scatter(dataPoints(finalClusters), ones(1, sum(finalClusters)) * 0.5 + 0.05 * rand(1, sum(finalClusters)), 50, 'r', 'o', 'filled');
scatter(dataPoints(~finalClusters), ones(1, sum(~finalClusters)) * 1.5 + 0.05 * rand(1, sum(~finalClusters)), 50, 'b', 'x', 'LineWidth', 2);
xlabel('X-axis (1D Data)');
ylabel('Cluster Assignment');
title('1D Data Clustering using Expectation-Maximization');
legend('Cluster A', 'Cluster B');
hold off;

fprintf('Final Parameters:\n');
fprintf('Cluster A -> Mean: %.2f, Variance: %.2f, Weight: %.2f\n', means(1), vars(1), mixWeightA);
fprintf('Cluster B -> Mean: %.2f, Variance: %.2f, Weight: %.2f\n', means(2), vars(2), mixWeightB);
