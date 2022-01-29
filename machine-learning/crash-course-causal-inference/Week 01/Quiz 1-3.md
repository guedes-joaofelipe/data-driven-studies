1. The Fundamental Problem of Causal Inference is that:

- treatment is typically not randomly assigned
- causal effects are only well defined for hypothetical interventions
- >we can only observe one potential outcome for each subject

2. Which of the following represents the causal effect of treatment on the treated?

- E(Y1)-E(Y0) 
- E(Y|A=1)-E(Y|A=0)
- >E(Y1|A=1)-E(Y0|A=1)
- E(Y1|A=1)-E(Y1|A=0)  

3. Which of the following represents the average causal effect for the population?

- E(Y1|A=1)-E(Y0|A=1)  
- >E(Y1)-E(Y0)  
- E(Y|A=1)-E(Y|A=0)  
- E(Y1|A=1)-E(Y1|A=0)  

4. Which assumption would be violated if the effectiveness of treatment on an individual depended on the treatment status of other individuals?

- Positivity
- >SUTVA
- Ignorability
- Consistency

5. Which assumption would be violated if we were interested in the causal effect of treatment for people age 40-80, but everyone over age 70 received the treatment?

- Consistency
- Ignorability
- SUTVA
- >Positivity

6. If the consistency assumption holds, then the observed outcome for a treated subject is equal to their potential outcome under that treatment.

- False
- >True

7. Which of the following can most easily be thought of as an intervention?

- Changing weight
- Changing ethnicity
- >Changing medication
- Changing blood pressure

8. Treatment assignment being ignorable given confounders, X, means:

- Treatment assignment is independent from X.
- Treatment assignment is independent from the observed outcomes, given X.
- >Within levels of X, treatment assignment is independent from the potential outcomes.

9. Computing means within levels of covariates and then combining these estimates is known as

- >standardization
- factor analysis
- calibration 