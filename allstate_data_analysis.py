import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.cross_validation import cross_val_score
from copy import deepcopy

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print train.shape
print test.shape
print train.describe()
print pd.isnull(train).values.any()

# print train.info()
cat_features = list(train.select_dtypes(include=['object']).columns)
print("Number of categorical variables are:", len(cat_features))
print cat_features

cont_features = [cont for cont in list(train.select_dtypes(
                 include=['float64', 'int64']).columns) if cont not in ['loss', 'id']]
print "Continuous: {} features".format(len(cont_features))

id_col = list(train.select_dtypes(include=['int64']).columns)
print "A column of int64: {}".format(id_col)

cat_uniques =[]
for cat in cat_features:
    cat_uniques.append(len(train[cat].unique()))

print cat_uniques

unique_val_categories = pd.DataFrame.from_items([('cat_names',cat_features),('unique_values',cat_uniques)])
print unique_val_categories.head()

fig,(ax1,ax2)= plt.subplots(1,2)
fig.set_size_inches(16,5)
ax1.hist(unique_val_categories.unique_values,bins=50)
ax1.set_title('Amount of categorical features with X distinct values')
ax1.set_xlabel('Distinct values in a feature')
ax1.set_ylabel('Features')

plt.figure(figsize=(16,8))
plt.plot(train['id'], train['loss'])
plt.title('Loss values per id')
plt.xlabel('id')
plt.ylabel('loss')
plt.legend()


print stats.mstats.skew(train['loss']).data
print stats.mstats.skew(np.log(train['loss'])).data

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches(16,5)
ax1.hist(train['loss'], bins=50)
ax1.set_title('Train Loss target histogram')
ax1.grid(True)
ax2.hist(np.log(train['loss']), bins=50, color='g')
ax2.set_title('Train Log Loss target histogram')
ax2.grid(True)
# plt.show()

train[cont_features].hist(bins=50,figsize=(16,12))
# plt.show()

for con in cont_features:
    print stats.mstats.skew(train[con]).data

plt.subplots(figsize=(16,9))
correlation_mat = train[cont_features].corr()
sns.heatmap(correlation_mat, annot=True)
plt.show()



