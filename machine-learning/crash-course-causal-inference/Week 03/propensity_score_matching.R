# Right heart catheterization (RHC) data 
# outcome: died
# treatment: right heart catheterization vs. not
# confounders: demographis, insurance, disease diagnoses, etc.

# load packages
library(tableone)
library(MatchIt)

# read in data
load(url("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/rhc.sav"))

# view data
View(rhc)

# create a data set with just these variables, for simplicity
ARF <- as.numeric(rhc$cat1=='ARF')
CHF <- as.numeric(rhc$cat1=='CHF')
Cirr <- as.numeric(rhc$cat1=='Cirrhosis')
Colcan <- as.numeric(rhc$cat1=='Colon Cancer')
Coma <- as.numeric(rhc$cat1=='Coma')
COPD <- as.numeric(rhc$cat1=='COPD')
lungcan <- as.numeric(rhc$cat1=='Lung Cancer')
MOSF <- as.numeric(rhc$cat1=='MOSF w/Malignancy')
sepsis <- as.numeric(rhc$cat1=='MOSF w/Sepsis')
female <- as.numeric(rhc$sex=='Female')
died <- as.numeric(rhc$death=='Yes')
age <- rhc$age
treatment <- as.numeric(rhc$swang1=='RHC')
meanbp1 <- rhc$meanbp1
aps <- rhc$aps

# new dataset
mydata <- cbind(ARF, CHF, Cirr, Colcan, Coma, COPD, lungcan, MOSF, sepsis, 
    meanbp1, aps, female, treatmnet, died)
mydata <- data.frame(mydata)

# fit a propensity score model using logistic regression
psmodel <- glm(
    treatment ~ ARF + CHF + Cirr + colcan + Coma + lungcan + MOSF + sepsis + age + female + meanbp1 + aps, 
    family = binomial(), # outcome is binary
    data=mydata
)

# show coefficients etc
summary(psmodel)

# create propensity score
pscore <- psmodel$fitted.values

# use matchit for propensity score, nearest neighbor matching
m.out <- matchit(
    treatment ~ ARF + CHF + Cirr + colcan + Coma + lungcan + MOSF + sepsis + age + female + meanbp1 + aps, 
    data=mydata,
    method="nearest"
)

summary(m.out)

# propensity score plots
plot(m.out, type="jitter")
plot(m.out, type="hist")

# do greedy matching on logit(PS)
psmatch <- Match(Tr=mydata$treatment, M=1, X=logit(pscore), replace=FALSE)
matched <- mydata[unlist(psmatch[c("index.treated", "index.control")]), ]
xvars <- c(ARF + CHF + Cirr + colcan + Coma + lungcan + MOSF + sepsis + age + female + meanbp1 + aps)
matchedtab1 <- CreateTableOne(vars=xvars, strata="treatment", data=matched, test=FALSE)

print (matchedtab1, smd=TRUE)


# re-do greedy matching on logit(PS) using caliper
psmatch <- Match(Tr=mydata$treatment, M=1, X=logit(pscore), replace=FALSE, caliper=.2)
matched <- mydata[unlist(psmatch[c("index.treated", "index.control")]), ]
xvars <- c(ARF + CHF + Cirr + colcan + Coma + lungcan + MOSF + sepsis + age + female + meanbp1 + aps)
matchedtab1 <- CreateTableOne(vars=xvars, strata="treatment", data=matched, test=FALSE)

print (matchedtab1, smd=TRUE)