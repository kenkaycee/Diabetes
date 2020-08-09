library(tidyverse); library(caret)
diabetes<- read.csv("diabetes.csv")
diabetes %>% str()
## convert aall predictors except age into factors 
diabetes<- diabetes %>% mutate_if(is.character, as.factor)
diabetes %>% str()
diabetes$class %>% table() %>% prop.table()*100
##  graphical representation 

diabetes %>% ggplot(aes(Obesity, fill=class))+
  geom_bar(position = "dodge")+
  facet_grid(.~AgeGroup)
table(diabetes$Age)
diabetes$AgeGroup[diabetes$Age<18]= "Child"
diabetes$AgeGroup[diabetes$Age >= 18 & diabetes$Age <=65]="Adult"
diabetes$AgeGroup[diabetes$Age > 65]= "Elderly"
diabetes$AgeGroup<- factor(diabetes$AgeGroup)
str(diabetes)

##  create train and test data set 
set.seed(100)
trainIndex<- createDataPartition(diabetes$class, p=.75, list = F)
train_diabetes<- diabetes[trainIndex,]
test_diabetes<- diabetes[-trainIndex,]

## compare frequencies of diagnosis in test and train data set against the original canceer dataset
train_prop<-round(prop.table(table(train_diabetes$class))*100,1)
test_prop<- round(prop.table(table(test_diabetes$class))*100,1)
original_prop<- round(prop.table(table(diabetes$class))*100,1) 
freq<- data.frame(cbind(original_prop,train_prop,test_prop))
colnames(freq)<- c("Original","Training", "Testing")
freq ## the frequencies are  similar

## parameter tuning 
fitCtrl<- trainControl(method = "repeatedcv", number = 10, repeats = 3)

##  KNN classification 
set.seed(100)
knnFit<- train(class~., data = train_diabetes, method="knn", preProcess=c("center","scale"), metric="Accuracy", tuneLength=17,
               trControl=fitCtrl)
knnFit
ggplot(knnFit)+
  scale_x_continuous(breaks = c(1:43)) # k=5 has highest accuracy on train data

## make predictions on test dataset 
knnPred<- predict(knnFit, test_diabetes)

##  build confusion matrix 
cmatKnn<- confusionMatrix(knnPred, test_diabetes$class, positive = "Positive")
cmatKnn # knn algorithm has 90% accuracy 

##  building classification tree using rpart 
set.seed(100)
rpartFit<- train(class~., data = train_diabetes, method="rpart", metric="Accuracy", tuneLength=17, trControl=fitCtrl)
##  view the classification tree
rpart.plot::rpart.plot(rpartFit$finalModel) 

##  make predicition on test data 
rpartPred<- predict(rpartFit, test_diabetes)
cmatRpart<- confusionMatrix(rpartPred, test_diabetes$class, positive = "Positive")
cmatRpart ## 87% Accuracy rate 

##  logistic regression 
set.seed(100)
logFit<-train(class~., data = train_diabetes, method="glm", family="binomial", metric="Accuracy", 
              tuneLength=17, trControl=fitCtrl)

# make predictions 
logPred<- predict(logFit, test_diabetes)
cmatLog<- confusionMatrix(logPred, test_diabetes$class, positive = "Positive")
cmatLog ## logistic regression has an accuracy rate of 93%

##  Random forest 
set.seed(100)
rfFit<- train(class~., data = train_diabetes, method="rf", metric="Accuracy", tuneLength=17, trControl=fitCtrl)
rfFit %>% plot() # mtry of 5 has highest accuracy 

## make predictiions on the test data set
rfPred<- predict(rfFit, test_diabetes)
cmatRf<- confusionMatrix(rfPred, test_diabetes$class, positive = "Positive")
cmatRf# randomforest has accuracy rate of 98%


gbmGrid<- expand.grid(.interaction.depth = (1:5) * 2,.n.trees = (1:10)*25, .shrinkage = c(0.01,0.05,0.1,0.5),
                      .n.minobsinnode=10)
set.seed(100)
gbmFit<- train(class~., data = train_diabetes, method="gbm", metric="Accuracy", 
               trControl=fitCtrl, tuneGrid=gbmGrid, verbose=FALSE, distribution="bernoulli",tuneLength=17)
gbmFit$finalModel

##  make a predition using test data set
gbmPred<- predict(gbmFit, test_diabetes)
cmatGbm<- confusionMatrix(gbmPred, test_diabetes$class, positive = "Positive")
cmatGbm ## accuracy of 96%

##  support vector machine 
set.seed(100)
svmFit<- train(class~., data = train_diabetes, method="svmRadial", trControl=fitCtrl, tuneLength=17, metric="Accuracy",
               preProcess=c("center", "scale"))
plot(svmFit, scales=list(log=2))

svmPred<- predict(svmFit, test_diabetes)
cmatSvm<- confusionMatrix(svmPred, test_diabetes$class, positive = "Positive")
cmatSvm ##  98% accuracy 

## compare performances of the models 
model_diff<- resamples(list(Knn=knnFit, LogisticReg=logFit, RpartTree=rpartFit, RandomForest=rfFit, GBM=gbmFit,
                            SVM= svmFit))
summary(model_diff)
dotplot(model_diff) # display the accuracy and kappa values for the different models (Radomforest, GBM and SVM have highest accuracy rates)
##  create a function that compares the performance of different models 
## function that to round values in a list if it is numeric 
round_num<- function(list){
  lapply(list, function(x){
    if(is.numeric(x)){
      x=round(x, 2) # round to 2 D.P
    }
    
  })
}

##  create a function that compares results of the models 

comp_summ<- function(cm, fit){
  summ<- list(TN= cm$table[1,1], #  True Negative
              TP= cm$table[2,2], #  True Positive
              FN= cm$table[1,2], #  False Negative
              FP= cm$table[2,1], #  False Positive
              Acc=cm$overall["Accuracy"], # Accuracy
              Sens=cm$byClass["Sensitivity"], # Sensitivity
              Spec=cm$byClass["Specificity"], # Specificity
              Prec=cm$byClass["Precision"], # Precision
              Recall= cm$byClass["Recall"], # Recall
              F1_Score=cm$byClass["F1"], #  F1 score
              PPV= cm$byClass["Pos Pred Value"], #  Positive predictive value
              NPV= cm$byClass["Neg Pred Value"] # Negative predictive value
  )
  round_num(summ) # rounds to 2 D.P
}

##  create a dataframe that stores the performance of the models

model_performance<- data.frame(rbind(comp_summ(cmatLog,logFit),
                                     comp_summ(cmatKnn,knnFit),
                                     comp_summ(cmatRpart,rpartFit),
                                     comp_summ(cmatSvm,svmFit),
                                     comp_summ(cmatRf,rfFit),
                                     comp_summ(cmatGbm,gbmFit)))
##  create names for rows in model performanc
rownames(model_performance)<- c("LogisticReg","KNN","RpartTree","SVM","RandomForest", "GBM")
model_performance


