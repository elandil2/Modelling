import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib


df = pd.read_excel('datas.xlsx')
features_df= pd.read_excel('datas.xlsx')


#Feature Select girmesini istemediğin özelliği sil -statik özellik olucak class
del features_df['target']
del features_df['last_new_job']
del features_df['gender']
del features_df['education_level']

#String kabul etmiyor class sütunu labellandı
le = LabelEncoder()

features_df['gender'] = le.fit_transform(df['gender'])
features_df['relevent_experience'] = le.fit_transform(df['relevent_experience'])
features_df['enrolled_university'] = le.fit_transform(df['enrolled_university'])
features_df['education_level'] = le.fit_transform(df['education_level'])
features_df['company_type'] = le.fit_transform(df['company_type'])
#features_df['mathscore'] = le.fit_transform(df['mathscore'])


#X - Y dizileri Oluştur //X : Feature için class olmayan datadaki değerler Y: önceki halinden class alır.
X = features_df.values
y = df['target'].values
#y = df['readingscore'].values
# y = df['writingscore'].values



# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_house_classifier_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.f" % mse)