set.seed(7)
#Loading libraries

library(randomForest)
library(caret)
library(e1071)

#Loading training and testing dataset

#Actual training_data contains the stock price for year 2017 but we excluded it in order to use year 2017 data for testing
train_data <- read.csv("~/R_PROG/nift_train.csv")
test_data=read.csv("~/R_PROG/nift_test.csv")

nf1=train_data[,-c(1,2)] ##our training data without column company_name and quarter

# #Regression with outliers### 

rg=lm(Closing_Price~.,data=nf1)
summary(rg)
# R-squred 65%

#Removing outliers based on scatter plots##

nf2=nf1[-which(nf1$P.E>100|(nf1$EPS>300 | nf1$EPS<0)|nf1$P.E>100|nf1$Turnover>8000
               |nf1$Debt_to_equity_ratio>8|nf1$P.B>20|(nf1$Current_ratio>6
                                                       | nf1$Quick_ratio>6)),]

tr1=nf2##our training data without outliers
ts1=test_data[,-c(1,2)]##our test data
rr=ts1[,1]

##regression without outliers##

rg2=lm(Closing_Price~.,data=nf2)
summary(rg2)
ppp=predict(rg2,ts1)

plot(rr,type="l",col='red')
points(ppp,type="l",col='blue')

### Scaling function... did not give much good result

scl=function(x){(x-min(x))/(max(x)-min(x))}
scd=apply(nf2[,-1],MARGIN=2,scl)
scd=as.data.frame(scd)
names(scd)=names(nf2[,-1])
scd$Closing_Price=nf2$Closing_Price

##run the below command to see regression on our scaled data

rg4=lm(Closing_Price~.,data=scd)
summary(rg4)


##Random forest

tr2=tr1
tr2$Closing_Price=as.factor(tr2$Closing_Price)
random_tree=randomForest(Closing_Price~.,data=na.omit(tr2),nodesize=25,ntree=100,importance=TRUE)
predict(random_tree,ts1[,-1])
varImpPlot(random_tree,type=2)



###cross validation and random forest########################

train_control1 <- trainControl(method="repeatedcv", number=4, repeats=2)
##below code takes some time to run.. don't panic
model1 <- train(Closing_Price~., data=na.omit(tr1), trControl=train_control1, method="rf")


pp=predict(model1,ts1)#predicted value of our test data
rr=ts1[,1]#actual test data

MSE1=sum((rr - pp)^2)/length(rr)
#24524.85
tp1=data.frame(pred=pp,actual=rr)##just a dataframe which contains predicted and actual value

###plotting graph###
plot(rr,type="l",col='red')## plot with actual value
points(pp,type="l",col='blue')## plot with predicted value

#Run it to see the best parameters for modelling
model1$results
model1$finalModel

##My error rate formula
kk=(abs(rr-pp)/rr)*100 # my error rate formula
accuracy_=100-(sum(kk)/38)
accuracy_

###########################################################################

##SVM without tuning

library(e1071)
train_control2 <- trainControl(method="repeatedcv", number=4, repeats=2)
model_svm <- train(Closing_Price~., data=na.omit(tr1), trControl=train_control2, method="svmRadial")

model_svm$results
model_svm$finalModel

pp_svm=predict(model_svm, ts1)
rr=ts1[,1]#actual test data

MSE2=sum((rr - pp_svm)^2)/length(rr)
#228507.3

#plot it
plot(rr,type="l",col='red')
points(pp_svm,type="l",col='blue')

#just use this to see the accuracy and compare it with that of above random forest model
kk_svm=(abs(rr-pp_svm)/rr)*100
accuracy_svm=100-(sum(kk_svm)/38)
accuracy_svm
#81.85

#################################################################################

##SVM with tuning parameters

grid_radial <- expand.grid(sigma = c(0,0.1, 0.25, 0.5, 0.75,0.9),C = c(0.25, 0.5, 0.75,1, 1.5, 2,5))
train_control3 <- trainControl(method="repeatedcv", number=4, repeats=2)
svm_rad <- train(Closing_Price~., data = na.omit(tr1), method = "svmRadial",
                    trControl=train_control3,
                    tuneGrid = grid_radial,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

p_rad=predict(svm_rad, ts1)
rr=ts1[,1]

MSE3=sum((rr - p_rad)^2)/length(rr
#54285.43
plot(rr,type="l",col='red')
points(p_rad,type="l",col='blue')

kk_svm_r=(abs(rr-p_rad)/rr)*100
accuracy_svm_r=100-(sum(kk_svm_r)/38)
accuracy_svm_r
#84.31
#Thus tuned-SVM gives the best model with lowets MSE of 54285.43
