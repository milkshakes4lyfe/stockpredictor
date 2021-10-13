## StockPredictionML.R ########################################################

rm(list = ls())
setwd("~/Chaitu/Python Projects/Stock Predictor")

###############################################################################
## Importing Libraries ########################################################

library(caret)
library(klaR)
library(dplyr)
library(cluster)
library(readxl)
library(Metrics)
library(adabag)
library(fastAdaboost)
library(nnet)

###############################################################################
## Extracting Data ############################################################

# Loading the stock data
stockData <- read.csv("StockCrossSectionalDataForTraining_07052021.csv")
# Emitting rows with null values
stockData <- na.omit(stockData)

# Renaming first column 
names(stockData)[1] <- "IndexColumn"

# Shuffle data frame
set.seed(420)
for (i in 1:25) {
  rows = sample(nrow(stockData))
}

stockData <- stockData[rows,]

# Creating new data frame for if return >= threshold
threshold = 0.025
stockData$ReturnOverThreshold <- ifelse(stockData$Return > threshold, 1, 0)

# Removing original Return column
stockData <- select(stockData,!IndexColumn)

###############################################################################
## Splitting Data into Train/Test #############################################

# Variable that determines whether to use external test data or not, if false it
# will split stockData into training and test data. 
boolUseExtraTestData = FALSE

if (boolUseExtraTestData) {
  # Assigning all stockData as training data
  train <- stockData
  # Assigning extra test cases as test data 
  test <- read.csv("ExtraTestCases.csv")
  test <- na.omit(test)
  test$ReturnOverThreshold <- ifelse(test$Return > threshold, 1, 0)
  
} else {
  # Splitting stockData into training and test data 
  n <- nrow(stockData) # tells the number of rows in the data frame
  fracTrain <- 0.70 # fraction of data that is training (rest is for validation)
  train_indices <- 1:round(fracTrain * n) # get train indices
  train <- stockData[train_indices, ] # get rows corresponding to train indices
  test_indices <- (round(fracTrain * n) + 1):n # get test indices
  test <- stockData[test_indices, ] # get rows corresponding to test indices
}

###############################################################################
## Linear Regression on Training/Validation Data ##############################

# Linear regression model on training dataset 
dataset <- select(train,!ReturnOverThreshold)
LinregModel <- lm(Return ~ .,data=dataset)

# Step-wise regression on the linear regression to fine tune the model
StepLinregModel <- step(LinregModel,test='F')
summary(StepLinregModel)

# Prediction performance metrics on training dataset 
rmse_linreg_train = rmse(train$Return,fitted(StepLinregModel))
# Converting linreg to classification by seeing if [predicted value > threshold] 
prediction <- ifelse(fitted(StepLinregModel) > threshold, 1, 0)
prediction_metrics_linreg_train <- confusionMatrix(as.factor(prediction),
                                                   as.factor(train$ReturnOverThreshold))
print(prediction_metrics_linreg_train)

# Prediction performance metrics on test dataset 
rmse_linreg_test = rmse(test$Return,predict(StepLinregModel,test))
# Converting linreg to classification by seeing if [predicted value > threshold] 
prediction <- ifelse(predict(StepLinregModel,test) > threshold, 1, 0)
prediction_metrics_linreg_test <- confusionMatrix(as.factor(prediction),
                                                  as.factor(test$ReturnOverThreshold))
print(prediction_metrics_linreg_test)

# Adding test data prediction results to data frame 
linreg_test_results = c(prediction_metrics_linreg_test$overall["Accuracy"],
                        prediction_metrics_linreg_test$byClass["Sensitivity"],
                        prediction_metrics_linreg_test$byClass["Recall"],
                        prediction_metrics_linreg_test$byClass["F1"])
test_df <- data.frame(linreg = linreg_test_results)

###############################################################################
## Logistic Regression on Training/Validation Data ############################

# Logistic regression on training dataset 
dataset <- select(train,!Return) 
LogitregModel <- glm(ReturnOverThreshold ~ .,data=dataset,family="binomial")

# Step-wise regression on the logistic regression to fine tune the model 
StepLogitregModel <- step(LogitregModel,test='F')
summary(StepLogitregModel)

# Prediction performance metrics on training dataset  
prediction <- predict(StepLogitregModel,dataset,type='response')
prediction <- ifelse(prediction > 0.5, 1, 0)  # A prediction value > 0.5 is considered 1, otherwise 0
prediction_metrics_logit_train <- confusionMatrix(as.factor(prediction), 
                                                  as.factor(dataset$ReturnOverThreshold))
print(prediction_metrics_logit_train)

# Prediction performance metrics on test dataset  
prediction <- predict(StepLogitregModel,test,type='response')  
prediction <- ifelse(prediction > 0.5, 1, 0)  # A prediction value > 0.5 is considered 1, otherwise 0
prediction_metrics_logit_test <- confusionMatrix(as.factor(prediction), 
                                                 as.factor(test$ReturnOverThreshold))
print(prediction_metrics_logit_test)

# Adding test data prediction results to data frame 
logit_test_results = c(prediction_metrics_logit_test$overall["Accuracy"],
                       prediction_metrics_logit_test$byClass["Sensitivity"],
                       prediction_metrics_logit_test$byClass["Recall"],
                       prediction_metrics_logit_test$byClass["F1"])
test_df$logit <- logit_test_results

###############################################################################
## Conditional Inference Decision Tree on Training/Validation Data ############

# Conditional inference decision tree on training dataset 
dataset <- select(train,!Return)
dataset$ReturnOverThreshold <- as.factor(dataset$ReturnOverThreshold)
set.seed(420)
ctreeModel <- train(ReturnOverThreshold ~ ., method= 'ctree', data=dataset,
                    trControl=trainControl(method='repeatedcv', number=5, repeats=5),
                    tuneGrid=expand.grid(mincriterion=0.95))
ctreeModel$results
plot(ctreeModel$finalModel)

# Prediction performance metrics on training dataset 
prediction <- predict(ctreeModel, dataset, type = "raw")
prediction_metrics_ctree_train <- confusionMatrix(prediction, dataset$ReturnOverThreshold)
print(prediction_metrics_ctree_train)

# Prediction performance metrics on test dataset   
testdata <- test
testdata$ReturnOverThreshold <- as.factor(testdata$ReturnOverThreshold)
prediction <- predict(ctreeModel, test, type = "raw")
prediction_metrics_ctree_test <- confusionMatrix(prediction, testdata$ReturnOverThreshold)
print(prediction_metrics_ctree_test)

# Adding test data prediction results to data frame 
ctree_test_results = c(prediction_metrics_ctree_test$overall["Accuracy"],
                       prediction_metrics_ctree_test$byClass["Sensitivity"],
                       prediction_metrics_ctree_test$byClass["Recall"],
                       prediction_metrics_ctree_test$byClass["F1"])
test_df$ctree <- ctree_test_results

###############################################################################
## KNN Classification on Training/Validation Data #############################

# KNN on training dataset 
dataset <- select(train,!Return)
dataset$ReturnOverThreshold <- as.factor(dataset$ReturnOverThreshold)
set.seed(420)
knnModel <- train(ReturnOverThreshold ~ ., data = dataset, method = "knn", 
                trControl = trainControl(method="repeatedcv", number=10, repeats=3), 
                preProcess = c("center","scale"), tuneLength = 30)

# Prediction performance metrics on training dataset 
prediction <- predict(knnModel, dataset)
prediction_metrics_knn_train <- confusionMatrix(prediction, dataset$ReturnOverThreshold)
print(prediction_metrics_knn_train)

# Prediction performance metrics on test dataset 
testdata <- test
testdata$ReturnOverThreshold <- as.factor(testdata$ReturnOverThreshold)
prediction <- predict(knnModel, test)
prediction_metrics_knn_test <- confusionMatrix(prediction, testdata$ReturnOverThreshold)
print(prediction_metrics_knn_test)

# Adding test data prediction results to data frame 
knn_test_results = c(prediction_metrics_knn_test$overall["Accuracy"],
                     prediction_metrics_knn_test$byClass["Sensitivity"],
                     prediction_metrics_knn_test$byClass["Recall"],
                     prediction_metrics_knn_test$byClass["F1"])
test_df$knn <- knn_test_results

###############################################################################
## AdaBoost (Boosted) Decision Tree on Training/Validation Data ###############

# AdaBoost decision tree on training dataset 
dataset <- select(train,!Return)
dataset$ReturnOverThreshold <- as.factor(dataset$ReturnOverThreshold)
set.seed(420)
train_control <- trainControl(method ='repeatedcv', number=5, repeats=3)
tune_grid <- expand.grid(nIter=8, method='adaboost')
adaboostModel <- train(ReturnOverThreshold~. , method='adaboost', data=dataset,
                       trControl=train_control, tuneGrid=tune_grid)

# Prediction performance metrics on training dataset 
prediction <- predict(adaboostModel, dataset, type = "raw")
prediction_metrics_adaboost_train <- confusionMatrix(prediction, dataset$ReturnOverThreshold)
print(prediction_metrics_adaboost_train)

# Prediction performance metrics on test dataset
testdata <- test
testdata$ReturnOverThreshold <- as.factor(testdata$ReturnOverThreshold)
prediction <- predict(adaboostModel, testdata, type = "raw")
prediction_metrics_adaboost_test <- confusionMatrix(prediction, testdata$ReturnOverThreshold)
print(prediction_metrics_adaboost_test)

# Adding test data prediction results to data frame 
adaboost_test_results = c(prediction_metrics_adaboost_test$overall["Accuracy"],
                          prediction_metrics_adaboost_test$byClass["Sensitivity"],
                          prediction_metrics_adaboost_test$byClass["Recall"],
                          prediction_metrics_adaboost_test$byClass["F1"])
test_df$adaboost <- adaboost_test_results

###############################################################################
## Bagged AdaBoost Decision Tree on Training/Validation Data ##################

# Bagged adaboost decision tree on training data
dataset <- select(train,!Return)
dataset$ReturnOverThreshold <- as.factor(dataset$ReturnOverThreshold)
set.seed(420)
train_control <- trainControl(method ='repeatedcv', number=5, repeats=3)
tune_grid <- expand.grid(mfinal=3, maxdepth=10)
adabagModel <- train(ReturnOverThreshold ~ ., method='AdaBag', data=dataset,
                     trControl=train_control, tuneGrid=tune_grid)

# Prediction performance metrics on training dataset
prediction <- predict(adabagModel, dataset, type = "raw")
prediction_metrics_adabag_train <- confusionMatrix(prediction, dataset$ReturnOverThreshold)
print(prediction_metrics_adabag_train)

# Prediction performance metrics on test dataset 
testdata <- test
testdata$ReturnOverThreshold <- as.factor(testdata$ReturnOverThreshold)
prediction <- predict(adabagModel, testdata, type = "raw")
prediction_metrics_adabag_test <- confusionMatrix(prediction, testdata$ReturnOverThreshold)
print(prediction_metrics_adabag_test)

# Adding test data prediction results to data frame 
adabag_test_results = c(prediction_metrics_adabag_test$overall["Accuracy"],
                        prediction_metrics_adabag_test$byClass["Sensitivity"],
                        prediction_metrics_adabag_test$byClass["Recall"],
                        prediction_metrics_adabag_test$byClass["F1"])
test_df$adabag <- adabag_test_results

###############################################################################
# Neural Network on Training/Validation Data ##################################

# Neural Network on training data 
dataset <- select(train,!Return)
dataset$ReturnOverThreshold <- as.factor(dataset$ReturnOverThreshold)
set.seed(420)
train_control <- trainControl(method ='repeatedcv', number=5, repeats=3)
tune_grid <- expand.grid(size=15, decay=5e-4)
nnetModel <- train(ReturnOverThreshold ~ ., method='nnet', data=dataset,
                   trControl=train_control, tuneGrid=tune_grid, maxit=200)

# Prediction performance metrics on training dataset
prediction <- predict(nnetModel, dataset)
prediction_metrics_nnet_train <- confusionMatrix(prediction, dataset$ReturnOverThreshold)
print(prediction_metrics_nnet_train)

# Prediction performance metrics on test dataset 
testdata <- test
testdata$ReturnOverThreshold <- as.factor(testdata$ReturnOverThreshold)
prediction <- predict(nnetModel, testdata, type = "raw")
prediction_metrics_nnet_test <- confusionMatrix(prediction, testdata$ReturnOverThreshold)
print(prediction_metrics_nnet_test)

# Adding test data prediction results to data frame 
nnet_test_results = c(prediction_metrics_nnet_test$overall["Accuracy"],
                      prediction_metrics_nnet_test$byClass["Sensitivity"],
                      prediction_metrics_nnet_test$byClass["Recall"],
                      prediction_metrics_nnet_test$byClass["F1"])
test_df$nnet <- nnet_test_results

###############################################################################
## Performance Analysis on Test Data ##########################################

# Classification algorithm performance on test data 
write.csv(test_df,file="Prediction_Algorithm_Result_on_TestData_Summary.csv")

###############################################################################
## Prediction on New Data #####################################################

newData <- read.csv("CrossSectionalData4Prediction.csv")

# Linear Regression prediction on 5 day return (Regression converted to Classification) 
newDataLinregPrediction <- predict(StepLinregModel,newData)
newDataLinregPrediction <- ifelse(newDataLinregPrediction > threshold, "Yes", "No") 

# Logistic Regression prediction on whether 5 day return above threshold (Classification)
newDataLogitregPrediction <- predict(StepLogitregModel,newData,type='response')
newDataLogitregPrediction <- ifelse(newDataLogitregPrediction > 0.5, "Yes", "No")

# Conditional Inference Decision Tree prediction on whether 5 day return above threshold (Classification)
temp <- predict(ctreeModel,newData,type="raw")
newDataCtreePrediction <- as.numeric(levels(temp))[temp]
newDataCtreePrediction <- ifelse(newDataCtreePrediction == 1, "Yes", "No")

# KNN prediction on whether 5 day return above threshold (Classification)
newDataKNNPrediction <- predict(knnModel,newData)
newDataKNNPrediction <- ifelse(newDataKNNPrediction == 1, "Yes", "No")

# AdaBoost Decision Tree prediction on whether 5 day return above threshold (Classification)
newDataAdaBoostPrediction <- predict(adaboostModel,newData,type="raw")
newDataAdaBoostPrediction <- ifelse(newDataAdaBoostPrediction == 1, "Yes", "No")

# Bagged AdaBoost Decision Tree prediction on whether 5 day return above threshold (Classification) 
newDataAdaBagPrediction <- predict(adabagModel,newData,type="raw")
newDataAdaBagPrediction <- ifelse(newDataAdaBagPrediction == 1, "Yes", "No")

# Neural Network prediction on whether 5 day return above threshold (Classification)
newDataNNetPrediction <- predict(nnetModel,newData,type="raw")
newDataNNetPrediction <- ifelse(newDataNNetPrediction == 1, "Yes", "No")

# Vector indicating whether stock has at least one yes prediction for return above threshold
AtLeastOneYes <- ifelse(newDataLinregPrediction == "Yes" |
                        newDataCtreePrediction == "Yes" | 
                        newDataLogitregPrediction == "Yes" |
                        newDataKNNPrediction == "Yes" |
                        newDataAdaBoostPrediction == "Yes" |
                        newDataAdaBagPrediction == "Yes" |
                        newDataNNetPrediction == "Yes", 
                        "Yes", "No")

# Creating dataframe summarizing prediction results
df <- data.frame(Ticker = newData$X,
                 Linreg = newDataLinregPrediction,
                 Logitreg = newDataLogitregPrediction,
                 Ctree = newDataCtreePrediction,
                 KNN = newDataKNNPrediction,
                 AdaBoost = newDataAdaBoostPrediction,
                 AdaBag = newDataAdaBagPrediction,
                 NNet = newDataNNetPrediction)
write.csv(df,file="PredictionsForReturnOnNewData_Clean.csv")