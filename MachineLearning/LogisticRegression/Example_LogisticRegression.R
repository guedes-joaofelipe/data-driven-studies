# Logistic Regression in R
# Author: Joao Felipe Guedes
# Credits: https://stats.idre.ucla.edu/r/dae/logit-regression/

# The dataset contains data on undergraduate students, such as
# GRE (Graduate Record Exam Scores), GPA (Grade Point Average),
# the prestige of admited institution and an Admision Variable, 
# representing the success/failure of students in a given institution.


# ===============================
# ===== Importing Libraries =====
# ===============================

library(aod);
library(ggplot2);



# =============================
# ===== Importing Dataset =====
# =============================

data <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv");

# Checking the first few rows of the data
head(data);

# Admit is a binary dependent variable that takes into account 3 
# features: GPA, GRE and Rank.
# GRE and GPA can be taken as continuous variables, whereas Rank is a 
# Categorical variable (Although it is numerical). 


# ==========================
# ===== Pre-processing =====
# ==========================

summary(data)

# Checking for null entries
sapply(data, function (x) sum(is.na(x))) # No null entries are found
# The xtabs( ) function allows you to create crosstabulations using formula style input.

# This can also be used to check null entries
xtabs(~admit+rank, data = data)

# Converting rank to a factor so as to indicate Rank should be 
# treated as a categorical variable
data$rank <- factor(data$rank)


# ===============================
# ===== Fitting Logit Model =====
# ===============================

# Sintax: glm(formula, family, data) ref: https://www.statmethods.net/advstats/glm.html
#     
# family = type of linear model (binomial == logit, gaussian == identity, poisson == log, etc.)
# formula = Target_variable ~ Feature_1 + Feature 2 + ...
# data = dataframe 
logit_model <- glm(admit ~ gre + gpa + rank, data = data, family = "binomial");

# ==================================
# ===== Evaluating the Fitting =====
# ==================================

summary(logit_model)
# Explanations on summary outputs can be found in 
# https://feliperego.github.io/blog/2015/10/23/Interpreting-Model-Output-In-R
# Estimate: can be seen as the slope of the curve Target x Feature. 
# For instance, the estimate -0.67 indicates that Admission is related to Rank2 as 
#   Admission = -0.67 * Rank2 + b
# 
# Z-value: measures ohw many sd our coefficient estimate is far from 0. Ideally, it is as 
# far away as possible from zero, meaning the feature and the target has a strong statistical 
# relationship


wald.test(b = coef(logit_model), Sigma = vcov(logit_model), Terms = 4:6)

sapply(data, function(x) length(unique(x)))
sapply(data, sd) # Standard deviation



new.data <- with(data, data.frame(gre = mean(gre), gpa = mean(gpa), rank = factor(1:4)))
new.data

new.data$rankP <- predict(logit_model, newdata = new.data, type = "response")
new.data

new.data.2 <- with(data, 
                   data.frame(gre = rep(seq(from = 200, to = 800, length.out = 100), 4), 
                              gpa = mean(gpa), 
                              rank = factor(rep(1:4, each = 100))))
new.data.3 <- cbind(new.data.2, 
                    predict(logit_model, 
                    newdata = new.data.2, 
                    type = "link", se = TRUE))


new.data.3 <- within(new.data.3, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})
head(new.data.3)

ggplot(new.data.3,aes(x = gre, y = PredictedProb)) 
       #  + geom_ribbon(aes(ymin = LL,ymax = UL, fill = rank), alpha = 0.8)  
       #  + geom_line(aes(colour = rank),
       # size = 1)