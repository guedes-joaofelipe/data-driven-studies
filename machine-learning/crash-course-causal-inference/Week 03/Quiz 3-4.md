1. The propensity score is:

- The risk of the outcome among controls
- >The probability of treatment given covariates
- The risk of the outcome among the treated


2. Trimming the tails involves:
 
- Using a logit transformation of the propensity
score before matching
- >excluding subjects who have extreme values of the propensity score
- not including highly skewed variables in the propensity score model


3. If the propensity score is exactly equal to 0 or 1 for some subjects, which causal assumption is violated?

- Consistency assumption
- >Positivity assumption


4. Propensity score matching involves the following steps, in order:
 
- >1. Estimate propensity score; 2. Check propensity score overlap; 3. Match on propensity score; 4. Check covariate balance
- 1. Estimate propensity score; 2. Check covariate balance; 3. Match on propensity score; 4. Check for propensity score overlap
- 1. Match on covariates; 2. Estimate propensity score; 3. Check covariate balance


5. If we use a caliper on the propensity score of 0.1, then:
 
- If a matched pair differs by more than 0.1, the larger propensity score will be truncated to make the difference equal to 0.1 
- >matches will never differ in the propensity score by more than 0.1
- matches will differ in the propensity score by exactly 0.1


