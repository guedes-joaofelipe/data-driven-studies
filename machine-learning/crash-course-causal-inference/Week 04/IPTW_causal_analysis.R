install.packages("tableone")
install.packages("Matching")
install.packages("MatchIt")
install.packages("ipw")
install.packages("survey")
install.packages("sandwich")

# Data loading
library(tableone)
library(Matching)
library(MatchIt)
library(ipw)
library(survey)
library(sandwich)
data(lalonde)

# Q1: What are the minimum and maximum weights?
# A: 1.01 and 40.1
xvars <- colnames(lalonde)
xvars <- xvars[xvars!='treat']
# propensity score model
psmodel<-glm(treat~age+educ+black+hispan+married+nodegree+re74+re75,
             family=binomial(link ="logit"),data=lalonde) 

# value of propensity score for each subject
ps <- predict(psmodel, type = "response")
lalonde$pscore <- ps

# create weights
weight<-ifelse(lalonde$treat==1,1/(ps),1/(1-ps))

# max and min of weights
print(max(weight)) 
print(min(weight))

# Q2: SMD after weighting
# 0.11
# apply weights to data
weighteddata<-svydesign(ids = ~ 1, data =lalonde, weights = ~ weight)
# weighted table
weightedtable <-svyCreateTableOne(vars = xvars, strata = "treat",
                                  data = weighteddata, test = FALSE)
print(weightedtable, smd = TRUE)

# Q3: ATE after weighting
# Est: 224.68 95% CI: (-1559.32, 2008.67)
msm <- (svyglm(re78 ~ treat, design = svydesign(~ 1, weights = ~ weight, data=lalonde)))
coef(msm)
confint(msm)

# Q4: ATE after weight truncating
# 486.93  (-1090.64, 2064.51) 
weightmodel <- ipwpoint(exposure = treat, family = "binomial", link = "logit", numerator = ~ 1, 
                        denominator = ~ age + educ + black + hispan + married + nodegree + re74 + re75, 
                        data = lalonde, trunc = 0.01)
msm_trunc <- (svyglm(re78 ~ treat, design = svydesign(~ 1, weights = ~ weightmodel$weights.trunc, data=lalonde)))
coef(msm_trunc)
confint(msm_trunc)
