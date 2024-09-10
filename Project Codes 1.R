#installing libraries
data("spam", package = "kernlab")
install.packages("rpart")
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
install.packages("caTools")
library(caTools)
install.packages("tree")
library(tree)
library(Metrics)
library(caret)
install.packages("randomForest")
library(randomForest)

#1. Consider the dataset called spam in the kernlab package.
## a) Using a seed value of 425, partition the dataset into training and test sets where 80% of goes 
#into the training set and 20% goes into the test set. Make sure that the proportion of classes 
#remains the same in both sets.

set.seed(425) #setting the seed value

split=sample.split(spam$type,SplitRatio=0.8)

summary(split)

spamtr=subset(spam,split==TRUE) #creating the training subset
spamte=subset(spam,split==FALSE) #creating the test subset

#Testing the proportion of the classes in the training and test sets
mean(as.numeric(spamtr$type)-1)
mean(as.numeric(spamte$type)-1)


## b) Using the rpart package and training set, determine the largest possible tree. How many leaf 
#nodes do exist in the tree?

##largest tree

#creating the largest tree with rpart
largesttreespam=rpart(type~.,data=spamtr,minsplit=2,minbucket=1,cp=0)

numofleafnodes =  printcp(largesttreespam)[nrow(printcp(largesttreespam)),2]+1
cat("Number of leaf nodes in the biggest tree is:", numofleafnodes) 

prp(largesttreespam,type=5,extra=101,nn=TRUE,tweak=1)


##c) Make predictions in the test set and report the accuracy, error rate, false positive rate, 
#false negative rate, and precision
predspam=predict(largesttreespam,newdata=spamte,type="class") #predictions
table(spamte$type,predspam)
cat("Accuracy = ", accuracy(actual = spamte$type,predicted = predspam), "\n") #accuracy
cat("Error rate = ", 1-accuracy(actual = spamte$type,predicted = predspam), "\n") #error rate
cat("False Positive Rate = ",table(spamte$type,predspam)[2,1]/(table(spamte$type,predspam)[2,1]+table(spamte$type,predspam)[2,2]), "\n") #false positive rate
cat("False Negative Rate = ",table(spamte$type,predspam)[1,2]/(table(spamte$type,predspam)[1,2]+table(spamte$type,predspam)[1,1]), "\n") #false negative rate
cat("Precision = ",table(spamte$type,predspam)[1,1]/(table(spamte$type,predspam)[1,1]+table(spamte$type,predspam)[2,1]), "\n")



##d) What is the size of the tree in terms of the number of leaf nodes which makes the crossvalidation (CV) 
#error the smallest? Note that rpart function provides this automatically. What 
#is the smallest the tree which has a CV error smaller than the smallest CV error plus one 
#standard deviation of the error? Call this last tree “opttree”.

#smallest cv error

opt_index=which.min(largesttreespam$cptable[, "xerror"]) #finding the optimum cp
opt_index=which.min(unname(largesttreespam$cptable[, "xerror"]))
cp_opt=largesttreespam$cptable[opt_index, "CP"]

treespam_opt=prune.rpart(tree = largesttreespam, cp = cp_opt) #pruning the largest tree until optimum tree is formed

#printcp(treespam_opt)
prp(treespam_opt,type=5,extra=101,nn=TRUE,tweak=1)
print(treespam_opt$cptable)

numofleafnodes_opt =  printcp(treespam_opt)[nrow(printcp(treespam_opt)),2]+1 #printing the leaf node number
cat("Number of leaf nodes is:", numofleafnodes_opt) #printing the leaf node number


cat("Smallest CV error + 1 sd = ", 0.19034 + 0.011019) #xerror value of the second tree
opttree=rpart(type~.,data=spamtr,cp=0.0013793) #cp giving the xerror = 0.201359
printcp(opttree)
prp(opttree,type=5,extra=101,nn=TRUE,tweak=1) #smallest tree (opttree)



##e) Make predictions on the test set with opttree and report the accuracy, error rate, false positive rate,
#false negative rate, and precision. Compare the result with part c)

predopttree=predict(opttree,newdata=spamte,type="class")
table(spamte$type,predopttree)
cat("Accuracy = ", accuracy(actual = spamte$type,predicted = predopttree), "\n") #accuracy
cat("Error rate = ", 1-accuracy(actual = spamte$type,predicted = predopttree), "\n") #error rate
cat("False Positive Rate = ",table(spamte$type,predopttree)[2,1]/(table(spamte$type,predopttree)[2,1]+table(spamte$type,predopttree)[2,2]), "\n") #false positive rate
cat("False Negative Rate = ",table(spamte$type,predopttree)[1,2]/(table(spamte$type,predopttree)[1,2]+table(spamte$type,predopttree)[1,1]), "\n") #false negative rate
cat("Precision = ",table(spamte$type,predopttree)[1,1]/(table(spamte$type,predopttree)[1,1]+table(spamte$type,predopttree)[2,1]), "\n")




## a. Partition the data set into training and test sets with 75% going into the training set by using 
#a seed value of 582.

toy=read.csv("toyotacorolla.csv")
set.seed(582)
train=sample(1:1436,1077)
toytr=toy[train,]
toyte=toy[-train,]

## b. Using the rpart package and training set, determine the tree which gives the smallest crossvalidation error? 
#How many leaf nodes do exist in this tree? Which attributes are the most important?

largesttreetoy=rpart(Price~.,data=toytr,minsplit=2, minbucket=1, cp=0)
opt_index_toy=which.min(largesttreetoy$cptable[, "xerror"])
opt_index_toy=which.min(unname(largesttreetoy$cptable[, "xerror"]))
cp_opt_toy=largesttreetoy$cptable[opt_index_toy, "CP"]
tree_opt_toy = prune.rpart(tree=largesttreetoy, cp=cp_opt_toy)
prp(tree_opt_toy,type=5,extra=101,nn=TRUE,tweak=1)
numofleafnodes_toy =  printcp(tree_opt_toy)[nrow(printcp(tree_opt_toy)),2]+1 #printing the leaf node number
cat("numofleafnodes_toy is = ", numofleafnodes_toy)
summary(tree_opt_toy)


##c. Make predictions in the test set and report the RMSE, MAE, and MAPE
toy_predict=predict(tree_opt_toy,newdata=toyte) 
rmse(actual = toyte$Price,predicted = toy_predict) 
mae(actual = toyte$Price,predicted = toy_predict) 
mape(actual = toyte$Price,predicted = toy_predict) 


##d. Using the randomForest package and training set, generate models by playing with “mtry”, 
#nodesize”, and “ntree” parameters. What parameter combination gives the smallest RMSE in 
#the test set?

# Define the parameter grid
mtry_vec <- seq(1, ncol(toytr)-1,1) # Number of variables to randomly sample for each tree
nodesize_vec <- seq(2, 20, by=2) # Minimum size of terminal nodes
ntree_vec <- c(100, 500, 1000) # Number of trees in the forest

# Initialize variables to store the results
best_rmse <- Inf
best_params <- NULL

# Loop through the parameter grid and train/test the models
for (mtry in mtry_vec) {
  for (nodesize in nodesize_vec) {
    for (ntree in ntree_vec) {
      
      # Train the model on the training set
      rf <- randomForest(Price ~ ., data=toytr, mtry=mtry, ntree=ntree, nodesize=nodesize, importance=TRUE)
      
      # Make predictions on the test set
      preds <- predict(rf, newdata=toyte)
      
      # Calculate the RMSE
      rmse <- sqrt(mean((preds - toyte$Price)^2))
      
      # Update the best RMSE and best parameter combination
      if (rmse < best_rmse) {
        best_rmse <- rmse
        best_params <- list(mtry=mtry, nodesize=nodesize, ntree=ntree)
      }
      
      # Print the progress
      cat("mtry=", mtry, ", nodesize=", nodesize, ", ntree=", ntree, ", RMSE=", rmse, "\n")
      
    }
  }
}

# Print the best parameter combination and the corresponding RMSE
cat("\nBest parameters:", paste(names(best_params), unlist(best_params), sep="="), "\n")
cat("Best RMSE:", best_rmse, "\n")



##e. Comment on which input attributes are most important in making predictions.

#create the best tree according to the previous function which tries various parameters and determines the lowest RMSE among them
rf_toy_imp=randomForest(Price~.,data=toytr,mtry=4,nodesize=16 ,ntree=500 ,importance=TRUE)

#find importance
round(importance(rf_toy_imp), 2)
varImpPlot(rf_toy_imp)


##f. Compare RMSE, MAE, and MAPE in the test set obtained by rpart and randomForest models.
#rpart predictions
rmse(actual = toyte$Price,predicted = toy_predict) #1473.07919854306
mae(actual = toyte$Price,predicted = toy_predict) #988.901161880602
mape(actual = toyte$Price,predicted = toy_predict) #0.102872690963842

#randomForest predictions

rf_predict = predict(rf_toy_imp,newdata=toyte) 
rmse(actual = toyte$Price,predicted = rf_predict) #1058.612
mae(actual = toyte$Price,predicted = rf_predict) #804.0104
mape(actual = toyte$Price,predicted = rf_predict) #0.08538448
