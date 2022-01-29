# Right heart catheterization (RHC) data 
# outcome: died
# treatment: right heart catheterization vs. not
# confounders: demographis, insurance, disease diagnoses, etc.

# load packages
library(tableone)
library(Matching)

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

# new dataset
mydata <- cbind(ARF, CHF, Cirr, Colcan, Coma, COPD, lungcan, MOSF, sepsis, female, died)
mydata <- data.frame(mydata)

# covariates we will use (shorter list than you would use in practice 
xvars <- c("ARF", "CHF", "Cirr", "colcan", "Coma", "lungcan", "MOSF", "sepsis", "age", "female", "meanbp1")

# look at table 1 on pre-matching
table1 <- CreateTableOne(vars=xvars, strata="treatment", data=mydata, test=FALSE)

# include standardized mean difference (SMD)
print (table1, smd=TRUE)

# do greeding matching on Malahanobis distance
greedymatch <- Match(Tr=treatment, M=1, X=mydata[xvars])
matched <- mydata[unlist(greedymatch[c("index.treated", "index.control")]), ]

matchedtab1 <- CreateTableOne(vars=xvars, strata="treatment", data=matched, test=FALSE)
print (matchedtab1, smd=TRUE)

# if we want a causal risk difference, we an carry out a paired t-test

# outcome analysis
y_trt <- matched$died[matched$treatment==1]
y_con <- matched$died[matched$treatment==0]

# pairwise difference 
diffy <- y_trt-y_con

# paired t-test
t.test(diffy)