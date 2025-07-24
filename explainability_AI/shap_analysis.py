import shap
import xgboost
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# Load and train a model
X, y = fetch_california_housing(return_X_y=True)
model = xgboost.XGBRegressor().fit(X, y)

# Explain predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Visualize for one prediction
shap.plots.waterfall(shap_values[0])

# Save the plot
plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=300)
plt.close()
