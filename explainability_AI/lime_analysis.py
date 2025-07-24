import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Train a model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier().fit(X, y)

# Use LIME
explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=['f1', 'f2', 'f3', 'f4'], class_names=['setosa', 'versicolor', 'virginica'], discretize_continuous=True)
exp = explainer.explain_instance(X[0], model.predict_proba)

# Get explanation figure
fig = exp.as_pyplot_figure()
fig.savefig("lime_explanation.png", bbox_inches='tight', dpi=300)
plt.show()
