#Downloaded dataset from kaggle: Drop-Clean- Normalized then;
Train model is x and test model is y. Set them as %30 test size and %70 train data set size. Used fit regression model for train and saved this model as dumb so we can use this model later.

Then, Loaded the trained model created with train_model.py and created a numpy array based on the model's feature importances, sorted the feature labels based on the feature importance rankings from the model, then printed each feature label to show results.

Finally applied 5 different classifiers to this model. These are ;
•	RandomForestClassifier
•	AdaBoostClassifier
•	DecisionTreeClassifier
•	KNeighborsClassifier
•	SVC
