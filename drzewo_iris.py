from sklearn.datasets import load_iris
from sklearn.tree import (DecisionTreeClassifier, export_graphviz)
from yellowbrick.model_selection import (ValidationCurve, LearningCurve)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

iris = load_iris()
X = iris.data[:,2:] # Długosc i szerokosc płatka
y = iris.target

iris_df = pd.DataFrame(data = iris.data, columns=iris.feature_names)    
iris_df.sample(10)

### Drzewo decyzyjne - max_depth = 3
tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X,y)
export_graphviz(tree_clf, out_file="C:/Users/dluko/Desktop/UM/iris_drzewo.dot",
                feature_names=iris.feature_names[2:],class_names=iris.target_names,
                rounded=True, filled=True)

### Krzywa weryfikacji
fig, ax = plt.subplots(figsize=(6,4))
vc_viz = ValidationCurve(RandomForestClassifier(n_estimators=100), param_name="max_depth",
                         param_range=np.arange(1,11), cv=10, n_jobs=-1)
vc_viz.fit(X,y)
ax.legend(("Ocena treningowa", "Ocena krzyżowa"),frameon=True)
vc_viz.ax.set(title="Krzywa weryfikacji klasyfikatora drzewa losowego", xlabel="max_depth",
              ylabel="Ocena")

### Krzywa uczenia
fig, ax = plt.subplots(figsize=(6,4))
lc3_viz = LearningCurve(RandomForestClassifier(n_estimators=100),cv=10)
lc3_viz.fit(X,y)
ax.legend(("Ocena treningowa","Ocena krzyżowa"), frameon=True)
lc3_viz.ax.set(title="Krzywa uczenia klasyfikatora drzewa losowego",
               xlabel="Liczba próbek", ylabel="Ocena")