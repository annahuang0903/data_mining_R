#STAT 5330 Data Mining Project 2 R Code
#Group 14
#Yufang Guo (yg4ek), Minzhi Hu (mh6nm), Yehan Huang (yh5sc), Zhiqiu Jiang (zj3av)

#input data
ghost.data<-read.csv("train.csv", header=TRUE)
test<-read.csv("submit.csv",header=TRUE)
str(ghost.data)
#delete variable "id"
ghost.data<-ghost.data[, -1]
#type is the response, categorical variable with 3 levels

################
#EDA
#quantitative variables
par(mfrow=c(2,2))
hist(ghost.data$bone_length)
summary(ghost.data$bone_length)
hist(ghost.data$rotting_flesh)
summary(ghost.data$rotting_flesh)
hist(ghost.data$hair_length)
summary(ghost.data$hair_length)
hist(ghost.data$has_soul)
summary(ghost.data$has_soul)
cor(ghost.data[,1:4])
pairs(ghost.data[,1:4])
#scatterplot matrix
library(car)
scatterplotMatrix(~type+bone_length+rotting_flesh+hair_length+has_soul+color, data=ghost.data)
#no interesting pattern 
#correlation matrix 
cor(ghost.data[, c(1:4)])
#the quantitative variables are not highly correlated with each other

#distribution of color 
table(ghost.data$color)
#n.max > 10* n.min, unbalanced data
library(tidyverse)
ggplot(aes(x=ghost.data$color),data=ghost.data)+geom_bar()
table<-table(ghost.data[,5:6])
result<-chisq.test(table)
result

#distribution of type 
table(ghost.data$type)
#okay

#############
#Multinomial regression
#fit multinomial logistic regression 
#set reference level
ghost.data$type2<-relevel(ghost.data$type, ref="Ghost")
ghost.big<-ghost.data
ghost.data<-ghost.data[, -6]
#set Ghost as the reference level 
library(nnet)
ghost.mlr<-multinom(type2~., data=ghost.data)
summary(ghost.mlr)
#ANOVA table 
Anova(ghost.mlr)
#color is insignificant
ghost.mlr2<-update(ghost.mlr, ~.-color, data=ghost.data)
summary(ghost.mlr2)
#compare 2 models 
anova(ghost.mlr2, ghost.mlr)
#insignificant p-value, ghost.mlr2 is better 
#see if there is any significant interaction 
ghost.mlr3<-multinom(type2~.*., data=ghost.data)
Anova(ghost.mlr3)
#add 5 interactions
ghost.mlr4<-multinom(type2~bone_length*rotting_flesh+hair_length*color+
                         has_soul+bone_length:has_soul+rotting_flesh:has_soul+rotting_flesh:color,
                     data=ghost.data)
Anova(ghost.mlr4)
#exclude color and 2 interactions 
ghost.mlr5<-multinom(type2~bone_length+rotting_flesh+hair_length+has_soul
                     +bone_length:has_soul, data=ghost.data)
Anova(ghost.mlr5)
#model comparison 
anova(ghost.mlr2, ghost.mlr5, ghost.mlr4, ghost.mlr3)
anova(ghost.mlr2, ghost.mlr4)
#the ANOVA test indicates that ghost.mlr4 is the best 

#check model assumptions 
#1. the sample size is large enough
#2. IIA assumption
library(mnlogit)
library(mlogit)
#build the same regression using mlogit 
#since hmftest() can only be applied to mlogit()
ghost.new<-mlogit.data(ghost.data, varying=NULL, choice="type2", shape="wide")
ghost.mlogit<-mlogit(type2~1|bone_length*rotting_flesh+hair_length*color+
                         has_soul+bone_length:has_soul+rotting_flesh:has_soul+rotting_flesh:color,
                     data=ghost.new, reflevel = "Ghost")
ghost.mlogit2<-mlogit(type2~1|bone_length*rotting_flesh+hair_length*color+
                          has_soul+bone_length:has_soul+rotting_flesh:has_soul+rotting_flesh:color,
                      data=ghost.new, alt.subset=c("Ghost","Ghoul"), reflevel = "Ghost")
hmftest(ghost.mlogit, ghost.mlogit2)
#IIA assumption is met
#We can use ghost.mlr4

#check model performance
#using repeated sub-sampling 
n.split<-20
aper<-c(rep(0, 20))
for(i in 1:n.split)
{
    train.ind<-sample(1:371, 297, replace=FALSE)
    ghost.train<-ghost.big[c(train.ind), ]
    ghost.test<-ghost.big[-c(train.ind),]
    ghost.tmlr<-multinom(type2~bone_length*rotting_flesh+hair_length*color+
                             has_soul+bone_length:has_soul+rotting_flesh:has_soul+rotting_flesh:color,
                         data=ghost.train)
    ghost.pred<-predict(ghost.tmlr,ghost.test,type="probs")
    for(j in 1:74)
    {
        if(which.max(ghost.pred[j,])==1)
        {
            ghost.test$predtype[j]<-"Ghost"
        }
        else if(which.max(ghost.pred[j,])==2)
        {
            ghost.test$predtype[j]<-"Ghoul"
        }
        else
        {
            ghost.test$predtype[j]<-"Goblin"
        }
    }
    ghost.prediction<-table(ghost.test$type, ghost.test$predtype)
    aper[i]<-1-(sum(diag(ghost.prediction))/74)
}
summary(aper)
par(mfrow=c(1,1))
hist(aper)
#misclassification rate for the multinomial logistic regression is
#about 0.29


################
#LDA
#with color
library(MASS)
#reread data 
ghost.data<-read.csv("train.csv", header=TRUE)
ghost.data<-ghost.data[, -1]
train.lda<-lda(type ~ ., data = ghost.data,CV=T)
#confusion matrix
lda.tab<-table(ghost.data$type,train.lda$class)
lda.tab
#classification error rate
cls_error_lda = 1 - mean(ghost.data$type == train.lda$class)
cls_error_lda
#0.2668464
#without color
ghost.data2<-ghost.data[,-5]
train.lda2<-lda(type ~ ., data = ghost.data2,CV=T)
#confusion matrix
lda.tab2<-table(ghost.data2$type,train.lda2$class)
lda.tab2
#classification error rate
cls_error_lda2 = 1 - mean(ghost.data2$type == train.lda2$class)
cls_error_lda2
#0.2506739


################
#QDA
#with color
train.qda<-qda(type ~ ., data = ghost.data,CV=T)
qda.tab<-table(ghost.data$type,train.qda$class)
qda.tab
#classification error rate
cls_error_qda = 1 - mean(ghost.data$type == train.qda$class)
cls_error_qda
#0.3072776
#without color
train.qda2<-qda(type ~ ., data = ghost.data2,CV=T)
qda.tab2<-table(ghost.data2$type,train.qda2$class)
qda.tab2
#classification error rate
cls_error_qda2 = 1 - mean(ghost.data2$type == train.qda2$class)
cls_error_qda2
#0.2506739


################
#KNN
library(class)
ghost.data<-read.csv("train.csv", header=TRUE)
ghost.data<-ghost.data[, -1]
#split training and testing
train.size <- 297
train.ind <- sample(1:371, size=train.size, replace=FALSE)
ghost.train <- ghost.data[train.ind, ]
ghost.test <- ghost.data[-train.ind,]
#using different values of k
knn.1 <-  knn(ghost.train[,1:4], ghost.test[,1:4], ghost.train[,6], k=1)
knn.5 <-  knn(ghost.train[,1:4], ghost.test[,1:4], ghost.train[,6], k=5)
knn.20 <- knn(ghost.train[,1:4], ghost.test[,1:4], ghost.train[,6], k=20)
table(knn.1,ghost.test[,6])
#classification error rate
cls_error_knn1 = 1 - mean(knn.1==ghost.test[,6])
cls_error_knn1
#0.3918919
table(knn.5,ghost.test[,6])
cls_error_knn5 = 1 - mean(knn.5==ghost.test[,6])
cls_error_knn5
#0.3378378
table(knn.20,ghost.test[,6])
cls_error_knn20 = 1 - mean(knn.20==ghost.test[,6])
cls_error_knn20
#0.3108108


################
#Tree-based models
train <- read.csv("train.csv")
names(train)

# delete the "id" column
train <- train[ , -1]
names(train)

# build a single tree
library(rpart)
p1 <- rpart(type~., data=train)
# variable relative importance
rpart:::importance(p1)
# plot the single tree 
plot(p1)
# add text to the tree plot
text(p1)


# bagging
library(ipred)
# build the bagging model based on 80% (=297) of the train dataset
set.seed(10)
train.size <- 297
train.index <- sample(1:371, size=train.size, replace=FALSE)
train.training <- train[train.ind, ]
train.testing <- train[-train.ind, ]
# build the model based on bagging approach
p2 <- bagging(type~., data=train.training, coob=T)
yhat.bag <- predict(p2, newdata = train.testing)
mean(p2$err)
table.bag<-table(yhat.bag, train.testing$type)
train.mis.bag<-1-mean(yhat.bag==train.testing$type)
mean(train.mis.bag)
summary(yhat.bag)
summary(train.testing$type)
summary(yhat.bag==train.testing$type)

# boosting
library (gbm)
# use entire train set for boosting fitting
boost.ghost =gbm(type~.,data=train, distribution="gaussian", n.trees =5000, interaction.depth =4, shrinkage = 0.001)
mean(boost.ghost$train.error)
summary(boost.ghost)


# random forest
library(randomForest)
p3 <- randomForest(type~., data=train, importance=T)
p3
plot(p3, main="")
legend("topright", colnames(p3$err.rate),col=1:4,cex=0.8,fill=1:4)
names(p3)
p3$confusion
# create a table for the confusion matrix
table.p3 <- data.frame(p3$confusion)
table.p3
randomForest:::importance(p3)
# export the table
# install.packages("rJava")
library(rJava)
library(xlsx)
write.xlsx(table.p3, "H:\\My Drive\\STAT5330\\HW\\HW2\\RandomForest_table.xlsx")
# calculate the misclassification error rate
1-(table.p3[1,1]+table.p3[2,2]+table.p3[3,3])/(sum(table.p3))
# the importance of variables in the random forest model
p3$importance
# for the column of "MeansDecreseGini" we can plot it out:
a <- p3$importance[,"MeanDecreaseGini"]
plot(a)
varImpPlot(p3)

# cross-validation
n.split <- 1000
# take 80% of the observations to set our training set
# take the rest 20% of the observations as our testing set
# 371*0.8=296.8
train.size <- 297
train.mis<-c(rep(0, 1000))
for(i in 1:n.split){
    train.ind <- sample(1:371, size=train.size, replace=FALSE)
    train.train <- train[train.ind, ]
    test.train <- train[-train.ind,]
    p4 <- randomForest(type~., data=train.train, importance=T)
    test.predict <- predict(p4, newdata=test.train)
    table.p4<-table(test.predict, test.train$type)
    train.mis[i]<-1-mean(test.predict==test.train$type)
}
hist(train.mis, main="Histogram of Misclassification Error Rate after Cross-Validation",
     xlab="Test Dataset Misclassification Error Rate",
     ylab="Frequency")
summary(train.mis)
write.csv(as.array(summary(train.mis)), file="cross_validation_table.csv")
mean(train.mis)


################
#SVM
#install.packages("e1071")
library(e1071)

#check correlation of each variable by plot
head(ghost.data)
plot(ghost.data[,1:4],col= ghost.data$type) 
plot(ghost_model_2,ghost.data,bone_length~has_soul)
#there is no significant relationship between each variable, and no independent variable can be distinguished.

#find the lowest misclassification rate of kernel
#type: 'C-classification'
#kernel: 'linear','polynomial','radial','sigmoid'
set.seed(1)
ghost_model<- svm(type~., data= ghost.data, type= 'C-classification', kernel = 'sigmoid', cross=10)
summary(ghost_model)#10-fold cross-validation on training data:Total Accuracy: 74.39353
ghost_pred <- predict(object = ghost_model, newdata = ghost.data[,1:5])
ghost_Freq <- table(ghost.data[,6], ghost_pred)
ghost_Freq
ghost_accuracy <- sum(diag(ghost_Freq))/sum(ghost_Freq)
1-ghost_accuracy#0.2398922
# we can change kernel to  'linear','polynomial','radial' or 'sigmoid' and get different results.
#The misclassification rate of kernel equals to 'linear','polynomial','radial','sigmoid'
#are about 0.2183288,0.3072776,0.2183288,0.2398922, respectively. 
#So we choose the lowest one, which is linear or radial.


# kernel equals to radial

#find best cost and gamma by tune function
tuned<-tune(svm,type ~., data =ghost.data, kernel= "radial",
            ranges=list(gamma = 10^(-10:2), cost = 10^(-3:3)))
summary(tuned)

#put best gamma and cost into model and do 10-fold cross validation 
set.seed(1)
ghost_model_2 <- svm(type~., data= ghost.data, type= 'C-classification', kernel = 'radial', gamma=0.001,cost=10, cross=10)
summary(ghost_model_2)#10-fold cross-validation on training data:Total Accuracy: 74.93261 
ghost_pred_2 <- predict(object = ghost_model_2, newdata = ghost.data[,1:5])
ghost_Freq_2 <- table(ghost.data[,6], ghost_pred_2)
ghost_Freq_2
ghost_accuracy_2 <- sum(diag(ghost_Freq_2))/sum(ghost_Freq_2)
1-ghost_accuracy_2#0.2345013
#the misclassification rate of kernel equal to 'radial' is about 0.2345013.


# kernel equals to linear
#find best cost by tune function
tuned<-tune(svm,type ~., data =ghost.data, kernel= "linear",
            ranges=list(cost  = 10^(-3:3)))
summary(tuned)

#put best cost into model and do 10-fold cross validation 
set.seed(1)
ghost_model_2 <- svm(type~., data= ghost.data, type= 'C-classification', kernel = 'linear', cost=0.01, cross=10)
summary(ghost_model_2)#10-fold cross-validation on training data:Total Accuracy: 75.74124
ghost_pred_2 <- predict(object = ghost_model_2, newdata = ghost.data[,1:5])
ghost_Freq_2 <- table(ghost.data[,6], ghost_pred_2)
ghost_Freq_2
ghost_accuracy_2 <- sum(diag(ghost_Freq_2))/sum(ghost_Freq_2)
1-ghost_accuracy_2#0.2291105
#the misclassification rate of kernel equal to 'linear' is about 0.2291105. The result is better than kernel equal to 'radial'.


#Prediction
test<-read.csv("submit.csv",header=TRUE)
ghost.test<-test[, -1]
type<-predict (object= ghost_model_2 , newdata =ghost.test)
test= data.frame(ghost.test, type)
write.csv(test, file = "Prediction2.csv", row.names = FALSE)




