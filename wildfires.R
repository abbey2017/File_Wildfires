### 15.062 Project Code by Abiodun Olaoye

## Title: Leveraging Data Mining Techniques to Predict Wildfires
rm(list = ls())
library(caret)
library(FNN)
library(class)
library(forecast)
library(neuralnet,nnet)
## Load Data
wildfire.data.df <- read.csv("forestfires.csv")

L <- length(wildfire.data.df[,1])
## Preview top of data
head(wildfire.data.df)


## Investigate Skewness of Variables 

# plot histogram of burned area in original units (ha)
hist(wildfire.data.df$area, col="red", ylab ="Frequency", xlab
     ="Burned area(hectares)", main = "Histogram of burned area in Original units (ha)")


## Now perform log transformation of variables to reduce skewness

# Burned area
wildfire.data.df$area.log <- log10(1 + wildfire.data.df$area)

# plot histogram of transformed burned area
hist(wildfire.data.df$area.log, col="darkgreen", ylab ="Frequency", xlab
     ="Burned area", main = "Histogram of log transformed burned area (log10(1+area))")



 # plot histogram of X coordinate 
 hist(wildfire.data.df$X, col="red", ylab ="Frequency", xlab
     ="X Coordinate", main = "Histogram of X Coordinate")

 # plot histogram of Y 
 hist(wildfire.data.df$Y, col="red", ylab ="Frequency", xlab
     ="Y Coordinate", main = "Histogram of Y Coordinate")
 
 # plot histogram of FFMC 
 hist(wildfire.data.df$FFMC, col="red", ylab ="Frequency", xlab
      ="FFMC Code", main = "Histogram of FFMC Code")
 
 # plot histogram of DMC 
 hist(wildfire.data.df$DMC, col="red", ylab ="Frequency", xlab
      ="DMC Code", main = "Histogram of DMC Code")
 
 # plot histogram of DC 
 hist(wildfire.data.df$DC, col="red", ylab ="Frequency", xlab
      ="DC Code", main = "Histogram of DC Code")
 
 # plot histogram of ISI Index 
 hist(wildfire.data.df$ISI, col="red", ylab ="Frequency", xlab
      ="ISI Index", main = "Histogram of ISI Index")
 
 # plot histogram of outside temperature (Celsius)
 hist(wildfire.data.df$temp, col="red", ylab ="Frequency", xlab
      ="Outside Temperature (oC)", main = "Histogram of Temperature (oC)")
 
 
 # plot histogram of relative humidity (%)
 hist(wildfire.data.df$RH, col="red", ylab ="Frequency", xlab
      ="Outside Relative Humidity (%)", main = "Histogram of Relative Humidity (%)")
 
 # plot histogram of outside wind speed (km/hr)
 hist(wildfire.data.df$wind, col="red", ylab ="Frequency", xlab
      ="Outside wind speed (km/hr)", main = "Histogram of wind speed (km/hr)")
 
 # plot histogram of outside rain (mm/m^2)
 hist(wildfire.data.df$rain, col="red", ylab ="Frequency", xlab
      ="Outside rain (mm/m^2)", main = "Histogram of rain (mm/m^2)")
 
 
 
 ## Create dummy variables to represent binary categorical variable form of burned area data (non-sever = 0, severe = 1)
 wildfire.data.df$area.cat <- ifelse(wildfire.data.df$area.log < 0.5, 0, 1)

## Next select variables which are the most significant predictors
 
 # Exclude month and date due to domain knowledge that does does not affect fire out break directly
 
 # Run correlation analysis after extracting numerical metrological variables
 numerical.data.id <- c(1,2,5,6,7,8,9,10,11,12,13,14)
 met.data.id <- c(9,10,11,12,14,15)
 corr.data <- cor(wildfire.data.df[,numerical.data.id])
 #Correlation between meteorological predictors
 corr.met.data <- cor(wildfire.data.df[,met.data.id][,-c(5,6)])


## Sample data to create training and validation partition 
set.seed(1)
train.id <- sample(1:L, 0.7*L)
valid.id <- setdiff(1:L,train.id)
 
train.df <- wildfire.data.df[train.id,met.data.id]
valid.df <- wildfire.data.df[valid.id,met.data.id]
 
## Normalize met data including output variable
norm.model <- preProcess(train.df, method = c("center","scale"))
norm.train.df <- predict(norm.model, train.df)
norm.valid.df <- predict(norm.model, valid.df)
 

#Summary Stat
Summ.stat <- data.frame(mean=sapply(wildfire.data.df[,numerical.data.id],mean,na.rm=TRUE), median=sapply(wildfire.data.df[,numerical.data.id],median,na.rm=TRUE),
                        min=sapply(wildfire.data.df[,numerical.data.id],min,na.rm=TRUE), max=sapply(wildfire.data.df[,numerical.data.id],max,na.rm=TRUE),
                        SD=sapply(wildfire.data.df[,numerical.data.id],sd,na.rm=TRUE))



                            ### PREDICTIVE MODELS

## (1) Create linear regression model 
lm.model  <- lm(area.log ~ ., data = norm.train.df[,-c(4,6)])

lm.valid.pred <- predict(lm.model,norm.valid.df[,-c(4,5,6)])
acc.lm.valid.df <- accuracy(lm.valid.pred,norm.valid.df[,5])


## (2) Create KNN model and obtain predictions to identify best K value
mae.valid.knn.df <- data.frame(k = seq(1,15,1), MAE = rep(0,15))
for(i in 1:15){
  set.seed(1)
 nn <- knn(norm.train.df[,-c(5,6)], norm.train.df[,-c(5,6)], cl = train.df[,5], k=i)
 nn <- as.numeric(as.character(nn))
 mae.valid.knn.df[i,2]  <- accuracy(nn,train.df[,5])[3]
}
mae.valid.knn.df

## Use the best K from above (K=6) to predit the burned area
set.seed(1)
knn.pred <- knn(norm.train.df[,-c(5,6)], norm.valid.df[,-c(5,6)], cl = train.df[,5], k=8, prob = TRUE)
knn.pred <- as.numeric(as.character(knn.pred))
opt.knn.acc.valid.df <- accuracy(knn.pred,valid.df[,5])


## (3) Neuralnet
norm.val <- preProcess(train.df, method = c("range"))
norm.2.train.df <- predict(norm.val, train.df)
norm.2.valid.df <- predict(norm.val, valid.df)


# Experiment to determine how optimal values of number of hidden layers and nodes per layer required for problem

    # Single Hidden Layer Experiment
acc.NN.valid.df <- data.frame(k = seq(1,10,1), MAE = rep(0,10))
for(i in 1:10) {
  set.seed(1)
  neuralnet.model <- neuralnet(area.log ~ temp+RH+wind+rain, data = norm.2.train.df, linear.output = T, hidden = i)
  valid.NN.pred <-   compute(neuralnet.model, norm.2.valid.df[, c("temp",  "RH","wind", "rain")])
  valid.NN.pred <- valid.NN.pred$net.result*range(train.df$area.log)[2] + min(train.df$area.log)
  acc.NN.valid.df[i,2] <- accuracy(norm.2.valid.df[,5],valid.NN.pred)[3]
}

   # Two Hidden Layers Experiment
acc.NN.valid.df <- data.frame(k = seq(1,10,1), MAE = rep(0,10))
for(i in 1:10) {
  set.seed(1)
neuralnet.model <- neuralnet(area.log ~ temp+RH+wind+rain, data = norm.2.train.df, linear.output = T, hidden = c(i,i))
valid.NN.pred <-   compute(neuralnet.model, norm.2.valid.df[, c("temp",  "RH","wind", "rain")])
valid.NN.pred <- valid.NN.pred$net.result*range(train.df$area.log)[2] + min(train.df$area.log)
acc.NN.valid.df[i,2] <- accuracy(norm.2.valid.df[,5],valid.NN.pred)[3]
}

 # Optimal Neural Network case
 set.seed(1)
 neuralnet.model2 <- neuralnet(area.log ~ temp+RH+wind+rain, data = norm.2.train.df, linear.output = T, hidden = c(3,3))
 valid.NN.pred2   <- compute(neuralnet.model2, norm.2.valid.df[, c("temp", "RH", "wind", "rain")])
 valid.NN.pred2   <- valid.NN.pred2$net.result*range(train.df$area.log)[2] + min(train.df$area.log)
 opt.acc.NN.valid.df <- accuracy(norm.2.valid.df[,5],valid.NN.pred2)
 opt.acc.NN.valid.df


## Compare predictive performance of models
acc.all.pred <- rbind(acc.lm.valid.df[,c(2,3)],opt.knn.acc.valid.df[,c(2,3)],opt.acc.NN.valid.df[,c(2,3)])
rownames(acc.all.pred) <- c("Multiple Linear Regression", "K-Nearest Neighbor","Neural Network")


                                   
                                             ### CLASSIFIERS

## Obtain classification in to severe and non-severe occurences using best K above (K = 6)

 ## (1) Logistic Regression
 glm.model <- glm(area.cat ~., data = train.df[,-5], family = "binomial")
 valid.glm.prob <- predict(glm.model,valid.df[,-c(5,6)], type = "response")

 # Initialize logistic regression prediction 
 valid.glm.pred <- valid.glm.prob

 # Obtain logistic regression classification using 0.5 cutoff threshold
 for( i in 1:length(valid.glm.prob)){
  ifelse(valid.glm.prob[i] >= 0.5,valid.glm.pred[i] <- 1,valid.glm.pred[i] <- 0)
 }
 confusionMatrix(valid.glm.pred, valid.df[,6], positive = "1")
 
 
 ## (2) KNN Classifier
 knnc.pred <- knn(norm.train.df[,-c(5,6)], norm.valid.df[,-c(5,6)], cl = train.df[,6], k=8,prob = TRUE)
 #nn <- as.numeric(as.character(nn))
 confusionMatrix(knnc.pred,valid.df[,6],positive = "1")
 
 
 ## (3) Neural Network Classifier
 norm.2.train.df$nonsevere <- ifelse(train.df$area.cat == 0, 1,0)
 norm.2.train.df$severe <- ifelse(train.df$area.cat == 0, 0, 1)
 #levels(norm.2.train.df$area.cat2) <- c("nonsevere", "severe")
 set.seed(1)
 neuralnet.model3 <- neuralnet(nonsevere+severe ~ temp+RH+wind+rain, data = norm.2.train.df, linear.output = T, hidden = 4, stepmax = 1e9)
 
 valid.NNC.pred  <- compute(neuralnet.model3,norm.2.valid.df[, c("temp","RH", "wind", "rain")])
                                                                  
 predict.class = apply(valid.NNC.pred$net.result,1,which.max)-1
 valid.df$area.cat2 <- ifelse(valid.df$area.cat =="1", "severe", "nonsevere")
 confusionMatrix(ifelse(predict.class =="1", "severe", "nonsevere"),valid.df$area.cat2)
 plot(neuralnet.model3)
 
 
 