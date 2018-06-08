import numpy as np
import pandas as pd
import cleanDataset

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt




                            # ------------------------------Start working on Cerevisiae--------------------------



data = np.array(pd.read_csv('cerevisiae.csv'))

dataset = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20]]
np.random.shuffle(dataset)


real_value_columns_cerevisiae = data[:, [7, 8, 9, 10, 11, 18, 19, 20]]

encoded_cerevisiae = cleanDataset.encodeRealColumns(real_value_columns_cerevisiae)

X = []
for i in range(6):
    X.append(dataset[:,i])

for i in range(8):
    for j in range(4):
        X.append(encoded_cerevisiae[j][i])

X_cerevisiae = np.asarray(X).transpose().astype(float)
y_cerevisiae = data[:, -1].astype(int)


                            #------------------------------Start working on Mikatae--------------------------



Mikatae = np.array(pd.read_csv('SMikatae.csv'))
nans = (np.isnan(Mikatae))
Mikatae[nans] = 0

real_value_columns_mikatae = Mikatae[:, [6, 7, 8, 9, 10, 11, 12, 13]]

encoded_mikatae = cleanDataset.encodeRealColumns(real_value_columns_mikatae)

X_mikatae = []

for i in range(6):
    X_mikatae.append(Mikatae[:, i])

for i in range(8):
    for j in range(4):
        X_mikatae.append(encoded_mikatae[j][i])

X_mikatae = np.asarray(X_mikatae).transpose()



                            # --------------------------------Start Prediction-----------------------------



BerNB = BernoulliNB(alpha = 0.1)

kfold = StratifiedKFold(n_splits = 10, shuffle=True)
prediction_cerevisiae = cross_val_score(BerNB, X_cerevisiae, y_cerevisiae, cv = kfold)

print('10-fold cross-validation prediction cerevisiae score:')
print(prediction_cerevisiae)

BerNB.fit(X_cerevisiae,y_cerevisiae)

predCer = BerNB.predict(X_cerevisiae)
prediction_mikatae = BerNB.predict(X_mikatae)

countZerosMikatae = 0
for i in prediction_mikatae:
    if i == 0:
        countZerosMikatae += 1

print('-----------------------------------------------')
print('predicted non essential mikatae:')
print(countZerosMikatae,'(',countZerosMikatae/len(prediction_mikatae), ')')
print('-----------------------------------------------')
print('predicted essential mikatae:')
print(len(prediction_mikatae) - countZerosMikatae , '(',(len(prediction_mikatae) - countZerosMikatae)/len(prediction_mikatae), ')')



                            # --------------------------------ROC function-----------------------------



predProbasCerevisiae = BerNB.predict_proba(X_cerevisiae)
predProbasMikatae = BerNB.predict_proba(X_mikatae)

print('-----------------------------------------------')
print('prediction probabilities mikatae:')
print(predProbasMikatae)

falsePositivesRate = dict()
truePositivesRate = dict()
roc_auc = dict()

falsePositivesRate, truePositivesRate, _ = roc_curve(y_cerevisiae, predProbasCerevisiae[:,1])
roc_auc = auc(falsePositivesRate, truePositivesRate)

plt.figure()

plt.plot(falsePositivesRate, truePositivesRate, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='darkgreen', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('R.O.C.')
plt.legend(loc="lower right")
plt.show()
