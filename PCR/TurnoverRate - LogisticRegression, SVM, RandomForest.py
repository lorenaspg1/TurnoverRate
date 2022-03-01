# Data cleaning and preparation
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Visualization
# ==============================================================================
import seaborn as sns

# Preprocessing and modeling
# ==============================================================================
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)

df = pd.read_csv('/Users/lorena/Git/PCA/HR.csv')



# 1. Rename columns
# ==============================================================================

df = df.rename(columns={'sales': 'department'})

df.isna().any()

df['department'] = np.where(df['department'] == 'IT', 'technical', df['department'])
df['department'] = np.where(df['department'] == 'support', 'technical', df['department'])

(df['department'].unique())  # the department column has the following categories

# 2. Data exploration
# ==============================================================================

print(df['left'].value_counts())
print(df.groupby('left').median())

print(df.groupby('department').mean())
print(df.groupby('left').mean())
print(df.groupby('salary').mean())

# 3. Data visualization
# ==============================================================================

pd.crosstab(df.department, df.left).plot(kind='bar')
plt.title('Turnover frequency for department')
plt.xlabel('Department')
plt.ylabel('Frequency of turnover')

table = pd.crosstab(df.salary, df.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')


# 4.Transform categorical variables
# ==============================================================================
# salary ordinal encoder

oe = OrdinalEncoder()
df['salary'] = oe.fit_transform(df[['salary']])

# department dummies

df_dummy = pd.get_dummies(df['department'], prefix='var')

df_joined = df.join(df_dummy)

df_joined.drop(columns=['department'], inplace=True)

X = df_joined.drop(columns=['left'])
y = df_joined['left']


# 5. Feature selection
# ==============================================================================

rfe = RFE(estimator=LogisticRegression(solver='lbfgs', max_iter=1000), n_features_to_select=11)
rfe = rfe.fit(X, y)
print(rfe.support_)
print(rfe.ranking_)

cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'var_RandD', 'var_hr', 'var_management', 'var_technical']

X = df_joined[['satisfaction_level', 'last_evaluation', 'number_project', 'time_spend_company', 'Work_accident',
               'promotion_last_5years', 'var_RandD', 'var_hr', 'var_management', 'var_technical']]
y = df_joined['left']
print(type(y))


# 6. Models
# ==============================================================================

# Logistic Regression Model
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(logreg)
print('Logistic Regression Accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))

# Random Forest
# ==============================================================================
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('Random Forest Accuracy: {:.3F}'.format(accuracy_score(y_test, rf.predict(X_test))))

# Support Vector Machine
# ==============================================================================
svc = SVC()
svc.fit(X_train, y_train)
print('Support Vector Machine Accuracy:{:.3F}'.format(accuracy_score(y_test, svc.predict(X_test))))

# 7. Checking Cross Validation
# ==============================================================================
# Cross validation ckecks if the results of a statistical analysis will generalize to an independent data set because they are independent of the partition
# The average accuracy remains very close to the Random Forest model accuracy; hence, we can conclude that the model generalizes well.

kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=7)
modelCV = RandomForestClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy : %.3f" % (results.mean()))
# The average accuracy remains very close to the Random Forest model accuracy; hence, we can conclude that the model generalizes well.


# 8. Precision and recall
# ==============================================================================

# We construct confusion matrix to visualize predictions made by a classifier and evaluate the accuracy of a classification

# Random Forest
# ==============================================================================
print(classification_report(y_test, rf.predict(X_test)))

y_pred = rf.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test)
sns.heatmap(forest_cm, annot=True, fmt='.2f', xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')

# Logistic Regression
# ==============================================================================
print(classification_report(y_test, logreg.predict(X_test)))
logreg_y_pred = logreg.predict(X_test)
logreg_cm = metrics.confusion_matrix(logreg_y_pred, y_test)
sns.heatmap(logreg_cm, annot=True, fmt='.2f', xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')

# Support Vector Machine
# ==============================================================================
print(classification_report(y_test, svc.predict(X_test)))
svc_y_pred = svc.predict(X_test)
svc_cm = metrics.confusion_matrix(svc_y_pred, y_test)
sns.heatmap(svc_cm, annot=True, fmt='.2f', xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Support Vector Machine')
plt.show()

# 9. The ROC curve
# ==============================================================================

# For logistic regression
roc_logistic_regression = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])

# For Random Forest
roc_random_forest = roc_auc_score(y_test, rf.predict(X_test))
random_fpr, random_tpr, random_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_logistic_regression)
plt.plot(random_fpr, random_tpr, label='Random Forest (area = %0.2f)' % roc_random_forest)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()

# 10. Feature Importance for Random Forest Model
# ==============================================================================

feature_labels = np.array(
    ['satisfaction_level', 'last_evaluation', 'number_project', 'time_spend_company', 'Work_accident',
     'promotion_last_5years', 'var_RandD', 'var_hr', 'var_management', 'var_technical'])
importance = rf.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] * 100.0)))
