from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from statsmodels.discrete.discrete_model import Logit
from statsmodels.discrete.discrete_model import LogitResults
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


def roc_curve(probabilities, labels):
    tprs, fprs, thres, i, t, f = [], [], [], 0, 0, 0
    p = sum(labels)
    n = len(labels) - p
    for (prob, lab) in sorted(zip(probabilities, labels), reverse=True):
        thres.append(prob)
        if lab == 0:
            f += 1

        else:
            t += 1
        tprs.append(t/p)
        fprs.append(f/n)
    return tprs, fprs, thres


df = pd.read_csv('data/loanf.csv')
y = (df['Interest.Rate'] <= 12).values
X = df[['FICO.Score', 'Loan.Length', 'Loan.Amount']].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

tpr, fpr, thresholds = roc_curve(probabilities, y_test)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of fake data")
plt.show()

grad = pd.read_csv('data/grad.csv')
ra = pd.crosstab(grad['rank'], grad['admit'])
ra['percent'] = ra[1]/(ra[0] + ra[1])
ra.percent.plot(kind='bar')
plt.ylabel("Acceptance Rate")
plt.show()

grad.gpa.hist()
plt.show()
grad.gre.hist()
plt.show()
# The GRE score is more noramlly distributed. A lot of people have high gpa

grad['ones'] = 1
y = grad.admit.values
X = grad[['gre', 'gpa', 'rank', 'ones']].values
logfit = sm.Logit(y, X)
results = logfit.fit()
print(results.summary())

logit = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
for train_index, va_index in KFold(n_splits=5).split(X_train):
    X_tr, X_va = X_train[train_index], X_train[va_index]
    y_tr, y_va = y_train[train_index], y_train[va_index]
    logit.fit(X_tr, y_tr)
    metrics.accuracy_score(y_test, logit.predict(X_test))
    metrics.precision_score(y_test, logit.predict(X_test))
    metrics.recall_score(y_test, logit.predict(X_test))
# predict scores.. but this runs 5 times so need to take the average

# pd.get_dummies(grad['rank'])
# original: 0.738, 0.625, 0.217, update 0.625, 1.0, 0.0625


tpr, fpr, thresholds = roc_curve(logit.predict_proba(X_test)[:, 1], y_test)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of fake data")
plt.show()

# Interpreting the beta coefficients with the Odds Ratio
logit = LogisticRegression()
logit.fit(X, y)

# Part 5: Predicted Probabilities
feat = pd.DataFrame(data=np.array([1, 2, 3, 4]), columns=['rank'])
feat['gpa'] = grad.gpa.mean()
feat['gre'] = grad.gre.mean()
feat['const'] = 1
feat = feat[['gre', 'gpa', 'rank', 'const']]

logit.predict_proba(feat.values)
plt.plot([1, 2, 3, 4], logit.predict_proba(feat.values)[:, 1])
plt.show()

odd = [p/(1-p) for p in logit.predict_proba(feat.values)[:, 1]]
plt.plot([1, 2, 3, 4], odd)
plt.show()

odd = [np.log(p/(1-p)) for p in logit.predict_proba(feat.values)[:, 1]]
plt.plot([1, 2, 3, 4], odd)
plt.show()
