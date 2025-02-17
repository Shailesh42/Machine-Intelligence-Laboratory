clc;
close all;

male = [6, 5.92, 5.58, 5.92; 190, 180, 170, 165; 12, 11, 12, 10];
female = [5, 5.5, 5.42, 5.75; 100, 150, 130, 150; 6, 8, 7, 9];

features = {'Height', 'Weight', 'Foot Size'};
maledata = zeros(3, 2);
femaledata = zeros(3, 2);

for i = 1:3
    maledata(i, 1) = mean(male(i, :)); 
    maledata(i, 2) = var(male(i, :)); 
    femaledata(i, 1) = mean(female(i, :)); 
    femaledata(i, 2) = var(female(i, :)); 
end

testdata = [6, 130, 8];
probmale = zeros(1, 3);
probfemale = zeros(1, 3);

for i = 1:3
    probmale(i) = (1 / (sqrt(maledata(i, 2)) * sqrt(2 * pi))) * ...
                  exp(-((testdata(i) - maledata(i, 1))^2) / (2 * maledata(i, 2)));
    probfemale(i) = (1 / (sqrt(femaledata(i, 2)) * sqrt(2 * pi))) * ...
                    exp(-((testdata(i) - femaledata(i, 1))^2) / (2 * femaledata(i, 2)));
end

disp('--- Mean and Variance for Male Data ---');
disp(array2table(maledata, 'VariableNames', {'Mean', 'Variance'}, ...
                 'RowNames', features));

disp('--- Mean and Variance for Female Data ---');
disp(array2table(femaledata, 'VariableNames', {'Mean', 'Variance'}, ...
                 'RowNames', features));

disp('--- Probabilities for Test Data ---');
for i = 1:3
    fprintf('Male probability for %s: %.6f\n', features{i}, probmale(i));
    fprintf('Female probability for %s: %.6f\n', features{i}, probfemale(i));
end

prior_prob_male = (length(maledata)) / (length(maledata) + length(femaledata));
prior_prob_female = (length(femaledata)) / (length(maledata) + length(femaledata));

disp('--- Prior Probabilities ---');
fprintf('Prior probability for Male: %.2f\n', prior_prob_male);
fprintf('Prior probability for Female: %.2f\n', prior_prob_female);

male_ev = prod(probmale) * prior_prob_male;
female_ev = prod(probfemale) * prior_prob_female;
sum =female_ev + male_ev;
male_ev = male_ev / sum;
female_ev = female_ev / sum;
disp('--- Posterior Probabilities ---');
fprintf('Posterior probability for Male: %.6f\n', male_ev);
fprintf('Posterior probability for Female: %.6f\n', female_ev);

if male_ev > female_ev
    disp('Result: The test data is classified as Male.');
else
    disp('Result: The test data is classified as Female.');
end

