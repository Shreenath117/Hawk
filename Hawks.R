# Case Study Solutions : Hawks.csv file

library(ROCR) #Creating ROC curve
library(PRROC) #Precision-recall curve
library(glmnet) #Lasso
library(tidyverse)
library(DT)
library(glmnet)
library(rpart)
library(rpart.plot)
library(caret)
library(knitr)
library(mgcv)
library(nnet)
library(NeuralNetTools)
library(knitr)
library(dplyr)
library(tidyr)
library(reshape2)
library(RColorBrewer)
library(GGally)
library(ggplot2)
library(caret)
library(glmnet)
library(boot)
library(verification)

#---------------------------------------------------------------------------------------------------------------
# Soln. to Question 1:
# Reading the CSV file and dropping the first column
data=read.csv('Hawks.csv')
# View the data loaded
data
# Dropping the first column which is nothing but the Serial number
data=data[2:20]
# View the dimensions (shape) of the data to be used for the analysis
dim(data)
# There are 908 rows and 19 columns

#---------------------------------------------------------------------------------------------------------------
# Soln. to Question 2:

# Measurements on Three Hawk Species
# The objective of case study is to predict which species out of the 3 hawk species given CH=Cooper's, RT=Red-tailed, SS=Sharp-Shinned based on the different predictor variables given in the dataset.


# Supervised machine learning: because we have labeled data here in the given dataset. 
# We can plan to construct a decision rule bsaed on which we can classify the sample points for the respective species

#---------------------------------------------------------------------------------------------------------------
# Soln. to Question 3:

summary(data)


# Check the datatypes
str(data)
sapply(data, class)

# We observe NA's in Wing, Weight, Culmen, Hallux, StandardTail, Tarsus, WingPitFat, KeelFat & Crop variables
# Year captured is between 1992 to 2003
# Wing	Length (in mm) of primary wing feather from tip to wrist it attaches to has the mean value of 772 with max being 2030
# Tail	Measurement (in mm) related to the length of the tail has the mean value of 199mm and max being 288mm
# Culmen measurement has a mean value of 21.8mm and a max value of 39.2mm
# Hallux lenth has a mean value of 26.41mm and a max value of 341.4mm

#-------------------------------------------------------------------------------------------------

# Soln. to Question 4:

for(i in 1:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
}


summary(data)


#-------------------------------------------------------------------------------------------------
# Soln. to Question 5:

# Levels of the prediction column
levels(data$Species)

freq_table <- table(data$Species)
freq_table

#-------------------------------------------------------------------------------------------------
# Soln. to Question 6:

## Histogram
hist(data$Wing)
hist(data$Tail)

#-------------------------------------------------------------------------------------------------
# Soln. to Question 7:

library(car)

scatterplot(y = data$Wing, x = data$Tail,
            main = 'Wing & Tail' ,
            ylab = 'Wing', xlab = 'Tail' ,
            regLine=list(method=lm, lty=1, lwd=2, col='red'),
            grid = FALSE)

#-------------------------------------------------------------------------------------------------
# Soln. to Question 8:

data1 <- subset(data, volunteer = 'no', select = c(Wing, Weight, Culmen, Hallux, Tail))

library(corrplot)
corrplot(cor(data1))

cor(data1)

# Wing , Weight , Culmen & Tail are Highly correlated

#-------------------------------------------------------------------------------------------------
# Soln. to Question 9:

## Box plot to understand how the distribution varies by class of species

data2 <- subset(data, volunteer = 'no', select = c(Wing, Weight, Culmen, Hallux, Tail,Species))

par(mfrow=c(1,5))
for(i in 1:5) {
  boxplot(data2[,i], main=names(data2)[i])
}


#-------------------------------------------------------------------------------------------------
# Soln. to Question 10:

library(ggthemes)
## Histogram
histogram <- ggplot(data=data, aes(x=Month)) +
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Culmen") +  
  ylab("Frequency") + 
  ggtitle("Histogram of Month")+
  theme_economist()
print(histogram)

#-------------------------------------------------------------------------------------------------
# Soln. to Question 11:

library(ggthemes)
## Histogram
histogram <- ggplot(data=data, aes(x=Culmen)) +
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Culmen") +  
  ylab("Frequency") + 
  ggtitle("Histogram of Culmen")+
  theme_economist()
print(histogram)


#-------------------------------------------------------------------------------------------------
# Soln. to Question 12:

## Faceting: Producing multiple charts in one plot
library(ggthemes)
facet <- ggplot(data=data, aes(Wing,Tail, color=Species))+
  geom_point(aes(shape=Species), size=1.5) + 
  geom_smooth(method="lm") +
  xlab("Wing") +
  ylab("Tail") +
  ggtitle("Faceting") +
  theme_fivethirtyeight() +
  facet_grid(. ~ Species) # Along rows
print(facet)

#-------------------------------------------------------------------------------------------------
# Soln. to Question 13:

# Creating a 2 way frequency table

freq_table <- table(data$Species,data$Age)
freq_table

par(mfrow=c(1,1))
# Draw a bar plot to the table created above: 
options(warn=-1)
barplot(freq_table, main="Species vs Age Distribution",xlab="Age",legend = rownames(freq_table), stacked=TRUE)

# CH has almost an equal distribution of Adult and Immature
# RT has higher immature ones compared to Adults...~3x
# SS has higher immature ones compared to Adults...~3x

#-------------------------------------------------------------------------------------------------
# Soln. to Question 14:

# Creating a 2 way frequency table

freq_table <- table(data$Species,data$Year)
freq_table

par(mfrow=c(1,1))
# Draw a bar plot to the table created above: 
options(warn=-1)
barplot(freq_table, main="Species vs Year Distribution",xlab="Year",legend = rownames(freq_table), stacked=TRUE)

# In 1994, shows the maximum  count followed by year 2000

#-------------------------------------------------------------------------------------------------
# Soln. to Question 15:


trainrows <- sample(nrow(data), nrow(data) * 0.70)
data.train <- data[trainrows, ]
data.test <- data[-trainrows,]

#-------------------------------------------------------------------------------------------------
# Soln. to Question 16:

data.train.glm0 <- glm(Species ~ Wing+Weight+Culmen+Hallux+Tail, family = binomial, data.train)

model1 <- multinom(Species ~ Wing+Weight+Culmen+Hallux+Tail, data.train)


#-------------------------------------------------------------------------------------------------
# Soln. to Question 17:
summary(model1)
summary(data.train.glm0)

#-------------------------------------------------------------------------------------------------
# Soln. to Question 18:

pred <- prediction(data.train.pred, data.train$yc)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

# Logistic Regression : 65%

#-------------------------------------------------------------------------------------------------
# Soln. to Question 19:

library(randomForest)

m3 <- randomForest(Species ~ Wing+Weight+Culmen+Hallux+Tail, data = data.train)

#-------------------------------------------------------------------------------------------------
# Soln. to Question 20:

summary(m3)

m3_fitForest <- predict(m3, newdata = data.test, type="prob")[,2]

#plot variable importance
varImpPlot(m3, main="Random Forest: Variable Importance")

#-------------------------------------------------------------------------------------------------
# Soln. to Question 21:

# Model Performance plot
plot(m3_perf,colorize=TRUE, lwd=2, main = "m3 ROC: Random Forest", col = "blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

m3_AUROC <- round(performance(m3_pred, measure = "auc")@y.values[[1]]*100, 2)
m3_AUROC
cat("AUROC: ",m3_AUROC)

# Random Forest : 73%
