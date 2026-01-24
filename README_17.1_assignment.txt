
Business objective of the task:

The primary business objective is to accurately identify and target customers who are most likely
subscribe to the bank products following a telephone marketing campaign.by analyzing customer demographics,
previous interactions, and campaign details, the goal is to develop predictive models that enable the bank
to optimize  marketing efforts, improve conversion rates, and allocate resource more efficiently. This will 
help the bank increase product adoption, reduce marketing costs, and enhance customer satisfaction by focusing
outreach on individuals with the highest likelihood of responding positively.


Baseline Model:

The most common baseline  for classification tasks is the accuracy of a simple model that always predicts 
the majority class (ie the class that occurs most frequently in the dataset)
For example, if 85% of customer in the dataset did not subscribe the product, a model that always predicts 
"no subscription" would have an accuracy of 85%
The classifiers that we would build should aim to beat the baseline accuracy. However if the data is imbalanced
(far more 'no' that 'yes' responses) also consider baseline metrics for precision, recall, and F1 score for the minority class ('yes)

Baseline Score: 

I find among the four models best Logistic Regression has the best Test Accuracy but at the cost of significant computational time.
With respect to low computational cost and reasonable Test Accuracy, KNN is the best model.

Score ; 0.8873458288821987

Logistic Regression:

Accuracy, Precision, recall, and F1 score are
0.9101723719349356
0.662771285475793
0.42459893048128344
0.5176010430247718



Comparison of 4 models:

 	Model 		 Train Accuracy  Test Accuracy  Fit Time (s)
0  Logistic Regression        0.911533       0.910172    163.492618
1                  KNN        0.919879       0.898155      0.047372
2        Decision Tree        1.000000       0.887837      0.377787
3                  SVM        0.928134       0.909080     41.203136


With tuning the parameters for each model, here is what I have got

Best Decision Tree Parameters: {'max_depth': 5, 'min_samples_split': 5}
Best Decision Tree CV Score: 0.91350531107739

Best KNN Parameters: {'n_neighbors': 10, 'weights': 'distance'}
Best KNN CV Score: 0.898300455235205

Best Logistic Regression Parameters: {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}
Best Logistic Regression CV Score: 0.9101972685887709


It is running for ever for SVM with the following parameters.

svm_params = {
    'C': [0.1, 1,10,100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
} 
svm_grid = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)
print("Best SVM Parameters:", svm_grid.best_params_)
print("Best SVM CV Score:",svm_grid.best_score_)


Colab link:

https://colab.research.google.com/drive/1w_0TwDsF8np_TNSKrSJqAOoi2drpwgeR#scrollTo=hsZIjg_wJyd-





