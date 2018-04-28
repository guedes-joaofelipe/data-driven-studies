# ===== Loading Dataset =====

attach(iris)
summary(iris)

# Transforming Dataset into Dataframe
df.iris <- as.data.frame(iris)
head(df.iris)

# ==== Preprocessing =====

library(ggplot2)
library(gridExtra)

# Data seems to be organized in consecutive species so I'll shuffle it
df.iris <- df.iris[sample(nrow(df.iris)),]
head(df.iris)

# Visualizing Variables
plt.sepalw.petalw <- ggplot(data = df.iris, aes(x = Sepal.Width, y = Petal.Width))+
  geom_point(aes(color = Species))

plt.sepall.petall <- ggplot(data = df.iris, aes(x = Sepal.Length, y = Petal.Length))+
  geom_point(aes(color = Species))

plt.sepalw.petall <- ggplot(data = df.iris, aes(x = Sepal.Width, y = Petal.Length))+
  geom_point(aes(color = Species))

plt.sepall.petalw <- ggplot(data = df.iris, aes(x = Sepal.Length, y = Petal.Width))+
  geom_point(aes(color = Species))

grid.arrange(plt.sepalw.petalw, plt.sepall.petalw, plt.sepalw.petall, plt.sepall.petall, nrow = 2, ncol = 2)

# Splitting Dataset Into Train and Validation Test
library(caTools)

set.seed(333)
split <- sample.split(df.iris, SplitRatio = 0.7)
data.train <- subset(df.iris, split == TRUE)
data.val <- subset(df.iris, split == FALSE)

# Splitting Data into Features and Target Variables
data.train.features <- data.train[-grep("Species", colnames(data.train))]
data.train.target <- data.train[grep("Species", colnames(data.train))]
data.val.features <- data.train[-grep("Species", colnames(data.val))]
data.val.target <- data.train[grep("Species", colnames(data.val))]

# Checking Variables Correlations
library(GGally)
ggcorr(data.train.features, label = TRUE, label_alpha = TRUE)

# Normalizing Data using z-score so as to have zero mean and unity variance
data.train.features <- scale(x = data.train.features, scale = TRUE, center = TRUE)
data.val.features <- scale(x = data.val.features, scale = TRUE, center = TRUE)

summary(data.train.features)

# ===== Training KNN =====
length(data.train.features)
length(data.train.target)

library(class)

knn.predictions <- knn(train = data.train.features, 
                      test = data.val.features, 
                      cl = data.train.target$Species, 
                      prob = TRUE, 
                      k = 3)

df.evaluations <- data.frame(
  pred = knn.predictions,
  true = data.val.target$Species)

table(df.evaluations)

