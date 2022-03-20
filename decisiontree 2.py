#Will they move to new job ?
"""A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses 
which conduct by the company. Many people signup for their training. Company wants to know which of these candidates are really wants to work for the company 
after training or looking for a new employment because it helps to reduce the cost and time as well 
as the quality of training or planning the courses and categorization of candidates."""




import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score


df = pd.read_excel('datas.xlsx')
features_df= pd.read_excel('datas.xlsx')


#Feature Select girmesini istemediğin özelliği sil -statik özellik olucak class
del features_df['target']
del features_df['gender']
del features_df['relevent_experience']
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





# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#print("They will move to new job accuracy:",metrics.accuracy_score(y_test, y_pred))


#entropy for error nodes
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
#print("They will move to new job accuracy entropied:",metrics.accuracy_score(y_test, y_pred))

from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
sc = StandardScaler()


X = df.drop(['target'], axis = 1)
y = df.target
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, random_state = 43, test_size = 0.3)

sc.fit(X_train, y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


rf = RandomForestClassifier()
ad = AdaBoostClassifier(base_estimator =rf)
dt = DecisionTreeClassifier()
kn = KNeighborsClassifier()
svc = SVC()

models = [rf,ad, dt, kn, svc]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mod = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = cross_val_score(model, X, y, cv=5).mean().round(3)
    accuracy = metrics.classification_report(y_test, y_pred)
    print(model, '\n', accuracy,'\n', 'mean_score:',scores, '\n' )