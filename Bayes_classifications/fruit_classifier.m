clc;
clear all;
close all;

fruit = [400, 350, 450, 500;   % Banana
         0, 150, 300, 300;     % Orange
         100, 150, 50, 200];  % Other

total = [500, 650, 800, 1000];
fruit_names = ["Banana", "Orange", "Other"];
features = ["Long", "Sweet", "Yellow"];

prob = zeros(1, 3);
for i = 1:3
    prob(i) = prod(fruit(i, :) ./ total);
end

prior_prob = sum(fruit, 2) ./ sum(fruit(:)); 
posterior_prob = prob .* prior_prob ;
posterior_prob = posterior_prob / sum(posterior_prob);
[max_posterior, index] = max(posterior_prob);
disp('--- Prior Probabilities ---');
for i = 1:3
    fprintf('%s: %.4f\n', fruit_names(i), prior_prob(i));
end

disp('--- Likelihood Probabilities ---');
for i = 1:3
    fprintf('%s (P(Features|Fruit)): %.4f\n', fruit_names(i), prob(i));
end
disp('--- Posterior Probabilities ---');
for i = 1:3
    fprintf('%s (P(Fruit|Features)): %.4f\n', fruit_names(i), posterior_prob(i));
end
fprintf('\nThe given data (%s) corresponds to: %s\n', strjoin(features, ', '), fruit_names(index));
