library(MASS)
library(ISLR)
library(leaps)
library(glmnet)
library(pls)
library(tidyverse)
library(ggplot2)
library(car)
library(caret)
library(elasticnet)
library(lars)

###Read in data
train.raw<-read.csv("train.csv",header = T)
test.raw<-read.csv("test.csv", header = T)
test.raw<-mutate(test.raw,FE=rep(0,290))
data.raw<-rbind(train.raw,test.raw)

###Avoid random changes
set.seed(37)

###Explore and transform variables
##EngDispl
summary(train.raw$EngDispl)
ggplot(aes(x=EngDispl,y=FE),data=train.raw)+geom_point()+geom_smooth(method=lm)
ggplot(aes(x=EngDispl,y=FE),data=train.raw)+geom_point()+geom_smooth()
#The relationship is negative, but looks curved
#Examine residual plot to find out
EngDispl.lm <- lm(FE ~ EngDispl, data=train.raw)
summary(EngDispl.lm)
#R^2 is 0.6486
ggplot(EngDispl.lm, aes(x=EngDispl, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#There is an obvious curved pattern
#Reciprocal transformation of EngDispl
train.raw<-mutate(train.raw,InvEngDispl=1/EngDispl)
EngDispl.lm2 <- lm(FE ~ InvEngDispl, data=train.raw)
summary(EngDispl.lm2)
#R^2 is 0.6866
ggplot(aes(x=InvEngDispl,y=FE),data=train.raw)+geom_point()+geom_smooth(method=lm)
ggplot(EngDispl.lm2, aes(x=InvEngDispl, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Quadratic transformation of EngDispl
EngDispl.lm3 <- lm(FE ~ EngDispl+I(EngDispl^2), data=train.raw)
summary(EngDispl.lm3)
#R^2 is 0.703
ggplot(EngDispl.lm3, aes(x=EngDispl, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Log transformation of EngDispl
train.raw<-mutate(train.raw,LogEngDispl=log(EngDispl))
EngDispl.lm4 <- lm(FE ~ LogEngDispl, data=train.raw)
summary(EngDispl.lm4)
#R^2 is 0.7014
ggplot(aes(x=LogEngDispl,y=FE),data=train.raw)+geom_point()+geom_smooth(method=lm)
ggplot(EngDispl.lm4, aes(x=LogEngDispl, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Decided to use log or quadratic transformation
#Change entire dataset
data.raw<-mutate(data.raw,LogEngDispl=log(EngDispl))

##NumCyl
summary(train.raw$NumCyl)
ggplot(aes(x=NumCyl,y=FE),data=train.raw)+geom_point()
#The relationship is negative, but looks curved
#Examine residual plot to find out
NumCyl.lm <- lm(FE ~ NumCyl, data=train.raw)
summary(NumCyl.lm)
#R^2 is 0.5645
ggplot(NumCyl.lm, aes(x=NumCyl, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#There is a curved pattern
#Reciprocal transformation of EngDispl
train.raw<-mutate(train.raw,InvNumCyl=1/NumCyl)
NumCyl.lm2 <- lm(FE ~ InvNumCyl, data=train.raw)
summary(NumCyl.lm2)
#R^2 is 0.5678
#Quadratic transformation of EngDispl
NumCyl.lm3 <- lm(FE ~ NumCyl+I(NumCyl^2), data=train.raw)
summary(NumCyl.lm3)
#R^2 is 0.6013
ggplot(NumCyl.lm3, aes(x=NumCyl, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Log transformation of EngDispl
train.raw<-mutate(train.raw,LogNumCyl=log(NumCyl))
NumCyl.lm4 <- lm(FE ~ LogNumCyl, data=train.raw)
summary(NumCyl.lm4)
#R^2 is 0.5932
#Decided not to transform because R^2 did not increase much

##NumGears
summary(train.raw$NumGears)
ggplot(aes(x=NumGears,y=FE),data=train.raw)+geom_point()
#The relationship is not strong.
#No transformation needed

##Transmission
#One-way ANOVA test
summary(train.raw$Transmission)
#Plot data
ggplot(train.raw, aes(x=Transmission, y=FE)) + 
    stat_summary(fun.y=mean, geom="point") +
    stat_summary(fun.y=mean, geom="line", aes(group=1)) +
    stat_summary(fun.data=mean_cl_normal, geom="errorbar", width=0.2)
#Residual plot
lt.mod <- lm(FE ~ Transmission, data=train.raw)
ggplot(lt.mod, aes(x=Transmission, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Levene's test
leveneTest(FE ~ Transmission, data=train.raw) #<0.05
#QQ plot
X <- data.frame(resid = residuals(lt.mod))
y <- quantile(X$resid, c(0.25, 0.75)) 
x <- qnorm(c(0.25, 0.75))  
slope <- diff(y)/diff(x) 
int <- y[1] - slope*x[1]   
ggplot(X, aes(sample = resid)) + stat_qq() + 
    geom_abline(intercept=int, slope=slope) 
#ANOVA test for FE differences in Transmission
lt.test <- aov(FE ~ Transmission, data=train.raw)
lt.test
summary(lt.test)
anova(lt.mod) #<0.05
#Transimission is significant, but equal variance does not hold
#Also need to find a way to group them Group transmission
#Fit linear regression using only Transmission
transmission.lm<-lm(FE~Transmission,data=train.raw)
summary(transmission.lm)
#Group them based on general transmission types
data.raw<-mutate(data.raw,TransmissionType=ifelse(Transmission=="A4" | Transmission=="A5" | Transmission=="A6" | Transmission=="A7"|Transmission=="AM6" | Transmission=="AM7", "A", 
                                         ifelse(Transmission=="AV" | Transmission=="AVS6", "AV",
                                                ifelse(Transmission=="M5" | Transmission=="M6" | Transmission=="Other", "M",
                                                       ifelse(Transmission=="S4" | Transmission=="S5" | Transmission=="S6" | Transmission=="S7"|Transmission=="S8", "S", "fail")))))%>%
    mutate(TransmissionType=as.factor(TransmissionType))

##AirAspirationMethod
#One-way ANOVA test
summary(train.raw$AirAspirationMethod)
#Plot data
ggplot(train.raw, aes(x=AirAspirationMethod, y=FE)) + 
    stat_summary(fun.y=mean, geom="point") +
    stat_summary(fun.y=mean, geom="line", aes(group=1)) +
    stat_summary(fun.data=mean_cl_normal, geom="errorbar", width=0.2)
#Residual plot
lt.mod <- lm(FE ~ AirAspirationMethod, data=train.raw)
ggplot(lt.mod, aes(x=AirAspirationMethod, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Levene's test
leveneTest(FE ~ AirAspirationMethod, data=train.raw) #>0.05
#QQ plot
X <- data.frame(resid = residuals(lt.mod))
y <- quantile(X$resid, c(0.25, 0.75)) 
x <- qnorm(c(0.25, 0.75))  
slope <- diff(y)/diff(x) 
int <- y[1] - slope*x[1]   
ggplot(X, aes(sample = resid)) + stat_qq() + 
    geom_abline(intercept=int, slope=slope) 
#ANOVA test for FE differences in AirAspirationMethod
lt.test <- aov(FE ~ AirAspirationMethod, data=train.raw)
lt.test
summary(lt.test) #0.00163
anova(lt.mod)
#Significance is large under 0.05, but not very large; may ignore this variable

##DriveDesc
summary(train.raw$DriveDesc)
#Plot data
ggplot(train.raw, aes(x=DriveDesc, y=FE)) + 
    stat_summary(fun.y=mean, geom="point") +
    stat_summary(fun.y=mean, geom="line", aes(group=1)) +
    stat_summary(fun.data=mean_cl_normal, geom="errorbar", width=0.2)
#Residual plot
lt.mod <- lm(FE ~ DriveDesc, data=train.raw)
ggplot(lt.mod, aes(x=DriveDesc, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Levene's test
leveneTest(FE ~ DriveDesc, data=train.raw) #<0.05
#QQ plot
X <- data.frame(resid = residuals(lt.mod))
y <- quantile(X$resid, c(0.25, 0.75)) 
x <- qnorm(c(0.25, 0.75))  
slope <- diff(y)/diff(x) 
int <- y[1] - slope*x[1]   
ggplot(X, aes(sample = resid)) + stat_qq() + 
    geom_abline(intercept=int, slope=slope) 
#ANOVA test for FE differences in DriveDesc
lt.test <- aov(FE ~ DriveDesc, data=train.raw)
lt.test
summary(lt.test) #<0.05
anova(lt.mod)
#Significance is large, but equal varaince does not hold
#Group some drives together
data.raw<- mutate(data.raw,DriveType=ifelse(DriveDesc=="ParttimeFourWheelDrive" | DriveDesc=="FourWheelDrive","FourWheel",
                                   ifelse(DriveDesc=="AllWheelDrive","AllWheel",
                                          ifelse(DriveDesc=="TwoWheelDriveFront","2WDFront",
                                                 ifelse(DriveDesc=="TwoWheelDriveRear","2WDRear","fail")))))%>%
    mutate(DriveType=as.factor(DriveType))

##CarlineClassDesc
summary(train.raw$CarlineClassDesc)
## Plot data
ggplot(train.raw, aes(x=CarlineClassDesc, y=FE)) + 
    stat_summary(fun.y=mean, geom="point") +
    stat_summary(fun.y=mean, geom="line", aes(group=1)) +
    stat_summary(fun.data=mean_cl_normal, geom="errorbar", width=0.2)
#Residual plot
lt.mod <- lm(FE ~ CarlineClassDesc, data=train.raw)
ggplot(lt.mod, aes(x=CarlineClassDesc, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Levene's test
leveneTest(FE ~ CarlineClassDesc,data=train.raw) #<0.05
## QQ plot
X <- data.frame(resid = residuals(lt.mod))
y <- quantile(X$resid, c(0.25, 0.75)) 
x <- qnorm(c(0.25, 0.75))  
slope <- diff(y)/diff(x) 
int <- y[1] - slope*x[1]   
ggplot(X, aes(sample = resid)) + stat_qq() + 
    geom_abline(intercept=int, slope=slope) 
#ANOVA test for FE differences in CarlineClassDesc
lt.test <- aov(FE ~ CarlineClassDesc, data=train.raw)
lt.test
summary(lt.test) #<0.05
anova(lt.mod)
#Significance is large, but equal varaince does not hold
#Group them based on size
data.raw<-mutate(data.raw,Size=ifelse(CarlineClassDesc=="CompactCars" | CarlineClassDesc=="MinicompactCars"|CarlineClassDesc=="SubcompactCars"|CarlineClassDesc=="2Seaters"|CarlineClassDesc=="SmallStationWagons","Small",
                                      ifelse(CarlineClassDesc=="MidsizeCars" | CarlineClassDesc=="SmallPickupTrucks2WD"|CarlineClassDesc=="SmallPickupTrucks4WD"|CarlineClassDesc=="Other","Medium",
                                             ifelse(CarlineClassDesc=="LargeCars" | CarlineClassDesc=="SpecialPurposeVehicleminivan2WD"|CarlineClassDesc=="SpecialPurposeVehicleSUV2WD"|CarlineClassDesc=="SpecialPurposeVehicleSUV4WD"|CarlineClassDesc=="StandardPickupTrucks2WD"|CarlineClassDesc=="StandardPickupTrucks4WD"|CarlineClassDesc=="VansCargoTypes"|CarlineClassDesc=="VansPassengerType","Large","fail"))))%>%
    mutate(Size=as.factor(Size))

##ModelYear
train.raw$ModelYear<-as.factor(train.raw$ModelYear)
summary(train.raw$ModelYear)
#Plot data
ggplot(train.raw, aes(x=ModelYear, y=FE)) + 
    stat_summary(fun.y=mean, geom="point") +
    stat_summary(fun.y=mean, geom="line", aes(group=1)) +
    stat_summary(fun.data=mean_cl_normal, geom="errorbar", width=0.2)
#Residual plot
lt.mod <- lm(FE ~ ModelYear, data=train.raw)
ggplot(lt.mod, aes(x=ModelYear, y=.resid)) + geom_point() + 
    geom_hline(yintercept=0, linetype="dashed")
#Levene's test
leveneTest(FE ~ ModelYear, data=train.raw) #<0.05
#QQ plot
X <- data.frame(resid = residuals(lt.mod))
y <- quantile(X$resid, c(0.25, 0.75)) 
x <- qnorm(c(0.25, 0.75))  
slope <- diff(y)/diff(x) 
int <- y[1] - slope*x[1]   
ggplot(X, aes(sample = resid)) + stat_qq() + 
    geom_abline(intercept=int, slope=slope) 
#ANOVA test for FE differences in ModelYear
lt.test <- aov(FE ~ ModelYear, data=train.raw)
lt.test
summary(lt.test)
anova(lt.mod)
#Significance is large, but equal varaince does not hold
data.raw<-mutate(data.raw,ModelYear=as.factor(ModelYear))

##TransLockup
train.raw$TransLockup<-as.factor(train.raw$TransLockup)
summary(train.raw$TransLockup)
group_by(train.raw,TransLockup)%>%summarise(mean(FE))
leveneTest(FE ~ TransLockup, data=train.raw) #>0.05
#t-test for two groups
t.test(FE ~ TransLockup, data=train.raw, var.equal=TRUE) #<0.05
#TransLockup is significant; only has 2 values, can treat as dummy variables
#Modify dataset
data.raw<-mutate(data.raw,TransLockup=as.factor(TransLockup))

##TransCreeperGear
train.raw$TransCreeperGear<-as.factor(train.raw$TransCreeperGear)
summary(train.raw$TransCreeperGear)
group_by(train.raw,TransCreeperGear)%>%summarise(mean(FE))
leveneTest(FE ~ TransCreeperGear, data=train.raw) #>0.05
#t-test for two groups
t.test(FE ~ TransCreeperGear, data=train.raw, var.equal=TRUE) #<0.05
#TransCreeperGear is significant; only has 2 values, may treat as dummy variables
#or ignore because of size imbalance
#Modify dataset
data.raw<-mutate(data.raw,TransCreeperGear=as.factor(TransCreeperGear))

##IntakeValvePerCyl and ExhaustValvePerCyl
train.raw$IntakeValvePerCyl<-as.factor(train.raw$IntakeValvePerCyl)
summary(train.raw$IntakeValvePerCyl)
group_by(train.raw,IntakeValvePerCyl)%>%summarise(mean(FE))
train.raw$ExhaustValvesPerCyl<-as.factor(train.raw$ExhaustValvesPerCyl)
summary(train.raw$ExhaustValvesPerCyl)
group_by(train.raw,ExhaustValvesPerCyl)%>%summarise(mean(FE))
#Test data
test.raw$IntakeValvePerCyl<-as.factor(test.raw$IntakeValvePerCyl)
summary(test.raw$IntakeValvePerCyl)
test.raw$ExhaustValvesPerCyl<-as.factor(test.raw$ExhaustValvesPerCyl)
summary(test.raw$ExhaustValvesPerCyl)
#Test data only has 1 and 2; may get rid of value of 0 and 3
train.raw.valve<- filter(train.raw,(IntakeValvePerCyl==1 | IntakeValvePerCyl==2) & (ExhaustValvesPerCyl==1 | ExhaustValvesPerCyl==2))
leveneTest(FE ~ IntakeValvePerCyl, data=train.raw.valve) #<0.05
t.test(FE ~ IntakeValvePerCyl, data=train.raw.valve, var.equal=F)
leveneTest(FE ~ ExhaustValvesPerCyl, data=train.raw.valve) #<0.05
t.test(FE ~ ExhaustValvesPerCyl, data=train.raw.valve, var.equal=F)
#Both are significant; may treat as dummy variables
#Modify dataset
data.raw<-filter(data.raw,(IntakeValvePerCyl==1 | IntakeValvePerCyl==2) & (ExhaustValvesPerCyl==1 | ExhaustValvesPerCyl==2))%>%
    mutate(IntakeValvePerCyl=as.factor(IntakeValvePerCyl),ExhaustValvesPerCyl=as.factor(ExhaustValvesPerCyl))
#Chi-square test on intake and exhaust
tb<-table(data.raw$IntakeValvePerCyl,data.raw$ExhaustValvesPerCyl)
chisq.test(tb) #<0.05
#intake and exhaust valves are not independent; may keep only one

##VarValveTiming
train.raw$VarValveTiming<-as.factor(train.raw$VarValveTiming)
summary(train.raw$VarValveTiming)
group_by(train.raw,VarValveTiming)%>%summarise(mean(FE))
leveneTest(FE ~ VarValveTiming, data=train.raw) #>0.05
#t-test for two groups
t.test(FE ~ VarValveTiming, data=train.raw, var.equal=TRUE)
#VarValveTiming is significant; only has 2 values, can treat as dummy variables
data.raw<-mutate(data.raw,VarValveTiming=as.factor(VarValveTiming))

##VarValveLift
train.raw$VarValveLift<-as.factor(train.raw$VarValveLift)
summary(train.raw$VarValveLift)
group_by(train.raw,VarValveLift)%>%summarise(mean(FE))
leveneTest(FE ~ VarValveLift, data=train.raw) #<0.05
#t-test for two groups
t.test(FE ~ VarValveLift, data=train.raw, var.equal=F)
#VarValveLift is significant; only has 2 values, can treat as dummy variables
data.raw<-mutate(data.raw,VarValveLift=as.factor(VarValveLift))
lapply(data.raw,class)

###Subset selection
train.sel<-dplyr::select(data.raw, FE,LogEngDispl,NumCyl,TransmissionType,NumGears,TransLockup,DriveType,ExhaustValvesPerCyl,Size,VarValveTiming,VarValveLift)%>%
    filter(FE!=0)
train.sel.x<-dplyr::select(train.sel,-FE)
train.sel.y<-dplyr::select(train.sel,FE)
regfit.lmnull <- lm(FE ~ 1, data = train.sel)
regfit.lmfull <- lm(FE ~ ., data = train.sel)

##Best subset
regfit.full <- regsubsets(FE ~ ., data = train.sel,nvmax = 15)
reg.summary <- summary(regfit.full)
reg.summary
#Does not treat categorical variables as whole
#Plot RSS, adjusted R2, Cp, and BIC for all of 
#the models to decide which model to select
par(mfrow = c(2, 2))
par(mar = rep(2, 4))
plot(reg.summary$rss, xlab = "num variables", ylab = "RSS", type = "l")
plot(reg.summary$adjr2, xlab = "num variables", ylab = "Adj. R2",type = "l")
maxpoint <- which.max(reg.summary$adjr2)
points(maxpoint, reg.summary$adjr2[maxpoint], col = "red", cex = 2,
       pch = 20)
plot(reg.summary$cp, xlab = "num variables", ylab = "Cp", type = "l")
maxpoint <- which.min(reg.summary$cp)
points(maxpoint, reg.summary$cp[maxpoint], col = "red", cex = 2,
       pch = 20)
plot(reg.summary$bic, xlab = "num variables", ylab = "BIC", type = "l")
maxpoint <- which.min(reg.summary$bic)
points(maxpoint, reg.summary$bic[maxpoint], col = "red", cex = 2,
       pch = 20)
bestsub.best <- 11
#Calculate train average prediction error for best model
#Prediction prediction error
lm.fit.best <- lm(FE ~ LogEngDispl+NumCyl+TransmissionType+NumGears+DriveType+ExhaustValvesPerCyl+Size+VarValveLift, data = train.sel)
bestsub.pred <- predict(lm.fit.best, train.sel)
mse.best = mean((train.sel.y - bestsub.pred)^2)
mse.best
#12.15598
rmse.best<-sqrt(mse.best)
rmse.best
#3.486543

##Forward, backward, hybrid stepwise
#Forward
regfit.forward <- step(regfit.lmnull, scope = list(lower = regfit.lmnull,
                                                   upper = regfit.lmfull), direction = "forward")
summary(regfit.forward)
fit.lm.f <- lm(FE~LogEngDispl+DriveType+Size+NumCyl+VarValveLift+TransmissionType+NumGears+ExhaustValvesPerCyl, data = train.sel)
pred <- predict(fit.lm.f, data.frame(train.sel.x))
mse.forward = mean((train.sel.y - pred)^2)
mse.forward
#12.15598
rmse.forward= sqrt(apply((train.sel.y-pred)^2,2,mean))
rmse.forward
#3.486543 
#Backward
regfit.backward <- step(regfit.lmfull, direction = "backward")
summary(regfit.backward)
fit.lm.b <- lm(FE ~ LogEngDispl+DriveType+Size+NumCyl+VarValveLift+TransmissionType+NumGears+ExhaustValvesPerCyl , data = train.sel)
pred <- predict(fit.lm.b, data.frame(train.sel.x))
mse.backward = mean((train.sel.y - pred)^2)
mse.backward
#12.15598
rmse.backward= sqrt(apply((train.sel.y-pred)^2,2,mean))
rmse.backward
#3.486543 
#Stepwise
regfit.step <- step(regfit.lmnull, scope = list(upper = regfit.lmfull),
                    direction = "both")
summary(regfit.step) #same

###Shrinkage method
train.shr<-dplyr::select(data.raw, FE,LogEngDispl,NumCyl,Transmission,AirAspirationMethod,NumGears,TransLockup,TransCreeperGear,DriveDesc,IntakeValvePerCyl,ExhaustValvesPerCyl,CarlineClassDesc,VarValveTiming,VarValveLift,ModelYear)%>%
    filter(FE!=0) 
train.shr.x<-dplyr::select(train.shr,-FE)
train.shr.y<-train.shr$FE

##Ridge regression
#K-fold to select best lambda
set.seed(37)
fitControl <- trainControl(method = "cv",number = 10)
lambdaGrid <- expand.grid(lambda = 10^seq(10, -2, length=100))
ridge <- train(FE~., data = train.shr,method='ridge',trControl = fitControl, preProcess=c('center', 'scale'))
ridge
predict(ridge$finalModel, type='coef', mode='norm')$coefficients[14,]
ridge.pred <- predict(ridge, train.shr.x)
sqrt(mean((ridge.pred - train.shr.y)^2))
#3.032881

##LASSO regression
lasso <- train(FE ~., train.shr,method='lasso',preProc=c('scale','center'),trControl=fitControl)
lasso
predict.enet(lasso$finalModel, type='coefficients', s=lasso$bestTune$fraction, mode='fraction')
lasso.pred <- predict(lasso, train.shr.x)
sqrt(mean((lasso.pred - train.shr.y)^2))
#3.037974

#Log transformation of FE
train.shr2<-dplyr::select(data.raw, FE,LogEngDispl,NumCyl,Transmission,AirAspirationMethod,NumGears,TransLockup,TransCreeperGear,DriveDesc,IntakeValvePerCyl,ExhaustValvesPerCyl,CarlineClassDesc,VarValveTiming,VarValveLift,ModelYear)%>%
    filter(FE!=0)%>%mutate(LogFE=log(FE))%>%dplyr::select(-FE)
train.shr.x2<-dplyr::select(train.shr2,-LogFE)
train.shr.y2<-train.shr2$LogFE
##Ridge regression
#K-fold to select best lambda
set.seed(37)
fitControl <- trainControl(method = "cv",number = 10)
lambdaGrid <- expand.grid(lambda = 10^seq(10, -2, length=100))
ridge <- train(LogFE~., data = train.shr2,method='ridge',trControl = fitControl, preProcess=c('center', 'scale'))
ridge
predict(ridge$finalModel, type='coef', mode='norm')$coefficients[14,]
ridge.pred.log <- predict(ridge, train.shr.x2)
sqrt(mean((ridge.pred.log - train.shr.y2)^2))
#0.081958
ridge.pred<-exp(ridge.pred.log)
sqrt(mean((ridge.pred - train.shr.y)^2))
#3.017068

##LASSO regression
lasso <- train(LogFE ~., train.shr2,method='lasso',preProc=c('scale','center'),trControl=fitControl)
lasso
predict.enet(lasso$finalModel, type='coefficients', s=lasso$bestTune$fraction, mode='fraction')
lasso.pred.log <- predict(lasso, train.shr.x2)
sqrt(mean((lasso.pred.log - train.shr.y2)^2))
#0.08221035
lasso.pred<-exp(lasso.pred.log)
sqrt(mean((lasso.pred - train.shr.y)^2))
#3.031003

###K-fold cross validation 
##Without log transformation
#Ridge
k=10
len<-floor(1154/k)
len
pd.error<-matrix(0,1,k)
for(i in 1:k){
    ind<-(len*i-len+1):(len*i)
    training<-train.shr[-ind,]
    testing<-train.shr[ind,]
    fitControl <- trainControl(method = "cv",number = 10)
    lambdaGrid <- expand.grid(lambda = 10^seq(10, -2, length=100))
    ridge <- train(FE~., data = training,method='ridge',trControl = fitControl, preProcess=c('center', 'scale'))
    ridge.pred <- predict(ridge, testing)
    pd.error[i]<-sqrt(mean((ridge.pred - testing$FE)^2))
}
pd.error
mean(pd.error)
#3.174719

#LASSO
k=10
len<-floor(1154/k)
len
pd.error2<-matrix(0,1,k)
for(i in 1:k){
    ind<-(len*i-len+1):(len*i)
    training<-train.shr[-ind,]
    testing<-train.shr[ind,]
    lasso <- train(FE ~., training,method='lasso',preProc=c('scale','center'),trControl=fitControl)
    predict.enet(lasso$finalModel, type='coefficients', s=lasso$bestTune$fraction, mode='fraction')
    lasso.pred <- predict(lasso, testing)
    pd.error2[i]<-sqrt(mean((lasso.pred - testing$FE)^2))
}
pd.error2
mean(pd.error2)
#3.165897

##With log transformation
#Ridge
k=10
len<-floor(1154/k)
len
pd.error<-matrix(0,1,k)
for(i in 1:k){
    ind<-(len*i-len+1):(len*i)
    training<-train.shr2[-ind,]
    testing<-train.shr2[ind,]
    ridge <- train(LogFE~., data = training,method='ridge',trControl = fitControl, preProcess=c('center', 'scale'))
    ridge.pred.log <- predict(ridge, testing)
    ridge.pred<-exp(ridge.pred.log)
    pd.error[i]<-sqrt(mean((ridge.pred - exp(testing$LogFE))^2))
}
pd.error
mean(pd.error)
#3.160526

#LASSO
k=10
len<-floor(1154/k)
len
pd.error2<-matrix(0,1,k)
for(i in 1:k){
    ind<-(len*i-len+1):(len*i)
    training<-train.shr2[-ind,]
    testing<-train.shr2[ind,]
    lasso <- train(LogFE ~., training,method='lasso',preProc=c('scale','center'),trControl=fitControl)
    lasso.pred.log <- predict(lasso, testing)
    lasso.pred<-exp(lasso.pred.log)
    pd.error2[i]<-sqrt(mean((lasso.pred - exp(testing$LogFE))^2))
}
pd.error2
mean(pd.error2)
#3.151742

###Predict test FE
#Using the LASSO model and log transformation of FE
lasso <- train(LogFE ~., train.shr2,method='lasso',preProc=c('scale','center'),trControl=fitControl)
lasso
coef<-predict.enet(lasso$finalModel, type='coefficients', s=lasso$bestTune$fraction, mode='fraction')
test<-dplyr::select(data.raw, FE,LogEngDispl,NumCyl,Transmission,AirAspirationMethod,NumGears,TransLockup,TransCreeperGear,DriveDesc,IntakeValvePerCyl,ExhaustValvesPerCyl,CarlineClassDesc,VarValveTiming,VarValveLift,ModelYear)%>%
    filter(FE==0) 
FE.log<-predict(lasso, test)
FE=exp(FE.log)
ID<-1158:1447
test.pred<-cbind(ID,FE)
coef1<-cbind(names(coef$coefficients),coef$coefficients)
write.csv(test.pred, file = 'pred1.csv', row.names = F)
write.csv(coef1,file="coef1.csv",row.names = F)

