import numpy as np
import joblib

# These are the feature labels from our data set
feature_labels = np.array(['city','gender','relevent_experience','enrolled_university','education_level','experience','company_size','company_type','last_new_job','training_hours','target'])

# Load the trained model created with train_model.py
model = joblib.load('trained_house_classifier_model.pkl')

# Create a numpy array based on the model's feature importances
importance = model.feature_importances_

# Sort the feature labels based on the feature importance rankings from the model
feauture_indexes_by_importance = importance.argsort()

# Print each feature label, from most important to least important (reverse order)
for index in feauture_indexes_by_importance:
    print("{} - {:.2f}%".format(feature_labels[index], (importance[index] * 100.0)))
    