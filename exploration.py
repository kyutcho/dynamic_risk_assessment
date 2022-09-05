#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#%%
X, y = make_classification(random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
# %%
clf.classes_
# %%
y
# %%
df = pd.read_csv("ingested_data/final_data.csv")
# %%
stats = df.describe(include=np.number).loc[["mean", "50%", "std"], :]
stats_dict = stats.to_dict(orient='list')

# %%
stats
# %%
stats_dict