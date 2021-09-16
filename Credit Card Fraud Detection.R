library(dplyr)#data manipulation
library(stringr)#for data manipulation
install.packages("caret")#for sampling
library(caret)

install.packages("caTools")#for training /test split
library(ggplot2)#for data visualization
library(corrplot)#for correlations
install.packages("corrplot")
install.packages("Rtsne")#for tsne plotting
library(caTools)#for smote implementation
library(Rtsne)#for tsne plotting 
install.packages("DMwR")#for smote implematation
library(DMwR)
install.packages("ROSE")
install.packages("SMOTE")
library(caTools)
library(ROSE)#for ROSE sampling
library(rpart)##for decision tree model
install.packages("Rborist")
library(Rborist)#for random tree model 
install.packages("XBoost")
library(xboost)#for xboost model

##To load data into R
head(archive,10)

str(archive)

summary(archive)
#checkinkg for missing values
colSums(is.na(archive)) #non of the variables have missing values 
#cheching for class imbalance
table(archive$Class)
#class imbalance in %
prop.table(table(archive$Class))*100
#visualizing the class imbalance
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))       
ggplot(data = archive, aes(x = factor(Class), 
                      y = prop.table(stat(count)), fill = factor(Class),
                      label = scales::percent(prop.table(stat(count))))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("no fraud", "fraud"))+
  scale_y_continuous(labels = scales::percent)+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Distribution of class labels") +
  common_theme 
#Distribution of variable "Time' by class
archive %>%
  ggplot(aes(x=Time,fill=factor(Class)))+geom_histogram(bins=100)+labs(x="Time in second since the first Transaction",y="No. of Transactions")+ggtitle("Distribution of time of Transaction by Class")+facet_grid(Class ~ .,scales = "free_y")+common_theme
#visualising box plots
ggplot(archive, aes(x = factor(Class), y = Amount)) + geom_boxplot() + 
  labs(x = 'Class', y = 'Amount') +
  ggtitle("Distribution of transaction amount by class") + common_theme
#correlation of anonymised variables and amount
correlations <- cor(archive[,-1],method="pearson")
correlations
corrplot(correlations, number.cex = .9, method = "circle", type = "full", tl.cex=0.8,tl.col = "black")
#We observe that most of the data features are not correlated. This is because before publishing, most of the features were presented to a Principal Component Analysis (PCA) algorithm. The features V1 to V28 are most probably the Principal Components resulted after propagating the real features through PCA. We do not know if the numbering of the features reflects the importance of the Principal Components.
#Visualising of Trnsaction Using tsne
#To try to understand the data better, we will try visualizing the data using t-Distributed Stochastic Neighbour Embedding, a technique to reduce dimensionality.
#To train the model, perplexity was set to 20.
#The visualisation should give us a hint as to whether there exist any "discoverable" patterns in the data which the model could learn. If there is no obvious structure in the data, it is more likely that the model will perform poorly.
#We use 1-% of the data to compute tsne

tsne_subset <- 1:as.integer(0.1*nrow(archive))
tsne <- Rtsne(archive[tsne_subset,-c(1, 31)], perplexity = 20, theta = 0.5, pca = F, verbose = F, max_iter = 500, check_duplicates = F)
classes <- as.factor(archive$Class[tsne_subset])
tsne_mat <- as.data.frame(tsne$Y)
ggplot(tsne_mat, aes(x = V1, y = V2)) + geom_point(aes(color = classes)) + theme_minimal() + common_theme + ggtitle("t-SNE visualisation of transactions") + scale_color_manual(values = c("#E69F00", "#56B4E9"))


#there appears to be a separation of transactions as the most fradulent transaction seem to appear at the edge of the blob data 

#Moddelling approach
#Here we will deal with clas  imbalance and try to balance the classess to improve our model performance 
#Remove Time Variable
archive<-archive[,-1]
head(archive)

#Change class variable to factor
archive$Class<-as.factor(archive$Class)
levels(archive$Class)<-c("Not_Fraud","Fraud")
head(archive)
#Scale Numeric Variables
archive[,-29]<-scale(archive[,-29])
head(archive)
#Split data into training and testin sets
set.seed(123)
split<-sample.split(archive$Class,SplitRatio=0.7)
train<-subset(archive,split=TRUE)
test<-subset(archive,split=FALSE)
##choosing a sampling technique -our dataset is unbalanceed and it is therfore neccessary that we create a balnce on the points of chsnge 
table(train$Class)#class ration inttially
# downsampling
set.seed(12345)
down_Train<-down_train <- downSample(x = train[, -ncol(train)],
                                     y = train$Class)
table(down_Train$Class)
#Upscaling
set.seed(12345)
up_Train<-up_train <- upSample(x = train[, -ncol(train)],
                               y = train$Class)
table(up_Train$Class)
#SMOTE-sYNTHETIC mINORITY SAMPLING TECHNIQUE
set.seed(12345)
smote_train <- SMOTE(Class ~ ., data  = train)
#Rose 
set.seed(12345)
rose_train <- ROSE(Class ~ ., data  = train)
table(smote_train$Class)
head(rose_train)
table(rose_train$Class)
#Decision trees in the orignal dataset (we are testng the our algorythmn on an original imbalanced dataset)

set.seed(1234)
orig_fit <- rpart(Class~.,data=train)
#Evaluating the original data model using test set
pred_orig<-predict(orig_fit,newdata=test,method="class")
pred_orig
#ROC curve 
roc.curve(test$Class,pred_orig[,2],plotit=T)
#Decision Tress on vario8us sampling techniques
set.seed(12345)
#Bulid a down sampled model
down_fit<-rpart(Class~.,data=down_Train)

set.seed(12345)
up_fit<-rpart(Class~.,data=up_Train)

set.seed(12345)
rose_fit<-rpart(Class~.,data = rose_train)

smote_fit<-rpart(Class~.,data = smote_train)

#AUC  on down sampled data
pred_down<-predict(down_fit,newdata=test)
print("fitting model to downsampled data")
roc.curve(test$Class,pred_down[,2],plotit=F)

#Area on up_sampled data
pred_up<-predict(up_fit,newdata = test)
print("fitting model on upsampled data")
roc.curve(test$Class,pred_up[,2],plotit=F)

#Area on smote data
pred_smote<-predict(smote_fit,newdata=test)
print("Fitting model to smote data")
roc.curve(test$Class,pred_smote[,2],plotit = F)
#AUC on ROSE data 
pred_rose<-predict(rose_fit,newdata=test)

#we see thst sll the sampling tchniques have yielded better auc scores than simple imbalancedndataset.We will test diffren tmodels using the upsampled technique as that has given the highest AUC score
#Models on upsampled data
glm_fit<-glm(Class~.,data=up_train,family="binomial")
pred_glm<-predict(glm_fit,newdata=test,type="response")
 roc.curve(test$Class,pred_glm,plotit = F)
#Random Forest
 x=up_train[,-30]
y=up_train[,30] 
 rf_fit<-Rborist(x,y,ntree=1000,minNode=20,maxLeaf=13)
rf_pred<-predict(rf_fit,test[,-30],ctgCensus="prob")
prob<-rf_pred$prob
roc.curve(test$Class,prob[,2],plotit=F)
#XGBoost
#Convert class label from factor to numeric
labels<-up_train$Class
y<-recode(labels,"Not_Fraud"=0,"Fraud"=1)
