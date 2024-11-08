# Supervised setting
## In supervised setting, the algorithm generates many of subset of features. Among those subset of features generated my the algorihtm is selected only one, the best one.
## How do we know which one is the best? At this stage, the fitness function come, k fold cross validation is implemented in this project with a set of classifiers such as Random Forest, Decision Tree, Support Vector Classifier, and Logistic Regression.
## Since this project is implemented using 5 fold cross validation, so each classifier is computed five times. Later, we got the mean of each classifier
## The fitness function will take mean of each classifier to find the mean of each classifier.
### Example, mean of Random Forest + mean of Decision Tree + mean of SVC + mean of Logistic Regression/4
## The subset feature which has the highest score(AUC) is selected
## Lastly, AUC score is calculated once again to ensure robustness

# Unsupervised setting

