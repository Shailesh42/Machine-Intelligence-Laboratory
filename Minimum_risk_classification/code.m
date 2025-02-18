clc;
clear;
close all;

% Prior Probabilities
P_Cancer = 0.20;      
P_NonCancer = 0.80;    

% Likelihood Probabilities (Conditional Probabilities)
P_Test = [0.10, 0.90;   % [P(Negative | Cancer), P(Negative | NonCancer)]
          0.90, 0.10];  % [P(Positive | Cancer), P(Positive | NonCancer)]

risk_matrix = [0, 10;  % [Risk (Chemo | Cancer), Risk (Chemo | NonCancer)]
               20, 0]; % [Risk (Medication | Cancer), Risk (Medication | NonCancer)]

P_Negative = P_Test(1,1) * P_Cancer + P_Test(1,2) * P_NonCancer;
P_Positive = P_Test(2,1) * P_Cancer + P_Test(2,2) * P_NonCancer;

P_Cancer_given_Negative = (P_Test(1,1) * P_Cancer) / P_Negative;
P_NonCancer_given_Negative = (P_Test(1,2) * P_NonCancer) / P_Negative;

P_Cancer_given_Positive = (P_Test(2,1) * P_Cancer) / P_Positive;
P_NonCancer_given_Positive = (P_Test(2,2) * P_NonCancer) / P_Positive;

Risk_Chemo_given_Negative = risk_matrix(1,1) * P_Cancer_given_Negative + risk_matrix(1,2) * P_NonCancer_given_Negative;
Risk_Medication_given_Negative = risk_matrix(2,1) * P_Cancer_given_Negative + risk_matrix(2,2) * P_NonCancer_given_Negative;

Risk_Chemo_given_Positive = risk_matrix(1,1) * P_Cancer_given_Positive + risk_matrix(1,2) * P_NonCancer_given_Positive;
Risk_Medication_given_Positive = risk_matrix(2,1) * P_Cancer_given_Positive + risk_matrix(2,2) * P_NonCancer_given_Positive;

% Decision rule: Choose the option with the minimum risk
if Risk_Chemo_given_Negative < Risk_Medication_given_Negative
    decision_negative = 'Chemotherapy';
else
    decision_negative = 'Medication';
end

if Risk_Chemo_given_Positive < Risk_Medication_given_Positive
    decision_positive = 'Chemotherapy';
else
    decision_positive = 'Medication';
end

disp('--- Minimum Risk Bayes Classifier ---');
disp('### Prior Probabilities ###');
fprintf('P(Cancer)     = %.2f\n', P_Cancer);
fprintf('P(Non-Cancer) = %.2f\n', P_NonCancer);


disp('### Likelihood Probabilities (Test Outcome Given Class) ###');
fprintf('P(Negative | Cancer)     = %.2f\n', P_Test(1,1));
fprintf('P(Negative | Non-Cancer) = %.2f\n', P_Test(1,2));
fprintf('P(Positive | Cancer)     = %.2f\n', P_Test(2,1));
fprintf('P(Positive | Non-Cancer) = %.2f\n', P_Test(2,2));

disp('### Marginal Probabilities of Test Outcomes ###');
fprintf('P(Negative Test) = %.2f\n', P_Negative);
fprintf('P(Positive Test) = %.2f\n', P_Positive);

disp('### Posterior Probabilities (Class Given Test Outcome) ###');
fprintf('P(Cancer | Negative)     = %.2f\n', P_Cancer_given_Negative);
fprintf('P(Non-Cancer | Negative) = %.2f\n', P_NonCancer_given_Negative);
fprintf('P(Cancer | Positive)     = %.2f\n', P_Cancer_given_Positive);
fprintf('P(Non-Cancer | Positive) = %.2f\n', P_NonCancer_given_Positive);

disp('### Expected Risks for Each Decision ###');
fprintf('Risk (Chemo | Negative Test)     = %.2f\n', Risk_Chemo_given_Negative);
fprintf('Risk (Medication | Negative Test) = %.2f\n', Risk_Medication_given_Negative);
fprintf('Risk (Chemo | Positive Test)     = %.2f\n', Risk_Chemo_given_Positive);
fprintf('Risk (Medication | Positive Test) = %.2f\n', Risk_Medication_given_Positive);

disp('### Final Decision Based on Minimum Risk ###');
fprintf('For a Negative test result, the decision is: %s\n', decision_negative);
fprintf('For a Positive test result, the decision is: %s\n', decision_positive);

