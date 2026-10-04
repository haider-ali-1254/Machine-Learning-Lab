import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("Car_Price_Prediction.csv").dropna()

target = next((c for c in ["Price", "Selling_Price", "Selling Price", "selling_price"] if c in data.columns), None)
if target is None:
    raise ValueError("Price column not found")

features = [c for c in ["Kms_Driven", "Year", "Engine", "Horsepower"] if c in data.columns]
if len(features) < 2:
    raise ValueError("Required car features not found")

X = data[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(data[target], errors="coerce")
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid]
y = y.loc[valid]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

print("LINEAR REGRESSION METRIC:")
print(f"R² Score: {r2_score(y_test, linear_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, linear_pred) ** 0.5:.4f}")

results = {}
for degree in [2, 3, 4]:
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results[degree] = model
    print(f"POLYNOMIAL REGRESSION DEGREE {degree}:")
    print(f"R² Score: {r2_score(y_test, prediction):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, prediction) ** 0.5:.4f}")

plot_feature = "Kms_Driven" if "Kms_Driven" in X.columns else X.columns[0]
values = X[plot_feature].sort_values().values
grid = pd.DataFrame([X_train.mean().values] * 300, columns=X.columns)
grid[plot_feature] = pd.Series(values).quantile([i / 299 for i in range(300)]).values

plt.figure()
for degree in [2, 3, 4]:
    plt.plot(grid[plot_feature], results[degree].predict(grid), label=f"Degree {degree}")
plt.scatter(X[plot_feature], y, alpha=0.5, label="Actual Data")
plt.xlabel(plot_feature)
plt.ylabel(target)
plt.title("Polynomial Regression")
plt.legend()
plt.show()
