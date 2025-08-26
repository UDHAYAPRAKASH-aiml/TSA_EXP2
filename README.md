# Ex.No: 02

## LINEAR AND POLYNOMIAL TREND ESTIMATION

## Developed By : udhaya prakash v

## Register Number:212224240177

## Date: 25-08-2025

---

### AIM:

To implement Linear and Polynomial Trend Estimation using Python.

---

### ALGORITHM:

1. Import necessary libraries (`pandas`, `numpy`, `matplotlib`, `sklearn`).
2. Load the dataset.
3. Convert the date column to datetime and extract the year.
4. Calculate the yearly average temperature.
5. Fit a **Linear Regression** model using least squares method.
6. Fit a **Polynomial Regression** model (degree 2).
7. Predict future temperature trends up to 2025.
8. Visualize the original data, linear trend, polynomial trend, and forecast.
9. End the program.

---

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("C:\\Users\\admin\\time series\\housing_price_dataset.csv")

# Group by YearBuilt to create a time series of average prices per year
price_series = data.groupby("YearBuilt")["Price"].mean().reset_index()

# Linear Regression (trend)
X = price_series[["YearBuilt"]]
y = price_series["Price"]

linear_model = LinearRegression()
linear_model.fit(X, y)
price_series["linear_trend"] = linear_model.predict(X)

# Polynomial Regression (Quadratic)
poly_degree = 2
X_poly = np.column_stack([X.values**i for i in range(1, poly_degree + 1)])
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
price_series["poly_trend"] = poly_model.predict(X_poly)

# ---- Plot Linear Trend ----
plt.figure(figsize=(12, 6))
plt.plot(price_series["YearBuilt"], price_series["Price"], 
         label="Original (Avg Price per Year)", color="blue")
plt.plot(price_series["YearBuilt"], price_series["linear_trend"], 
         label="Linear Trend", color="red", linestyle="--", linewidth=2)
plt.title("Linear Trend Estimation")
plt.xlabel("Year Built")
plt.ylabel("Average Price")
plt.legend()
plt.show()

# ---- Plot Polynomial Trend ----
plt.figure(figsize=(12, 6))
plt.plot(price_series["YearBuilt"], price_series["Price"], 
         label="Original (Avg Price per Year)", color="blue")
plt.plot(price_series["YearBuilt"], price_series["poly_trend"], 
         label="Polynomial Trend (Quadratic)", color="green", linestyle="-", linewidth=2)
plt.title("Polynomial Trend Estimation (Quadratic)")
plt.xlabel("Year Built")
plt.ylabel("Average Price")
plt.legend()
plt.show()

# Print coefficients to check overlap
print("Linear coefficients:", linear_model.coef_, "Intercept:", linear_model.intercept_)
print("Polynomial coefficients:", poly_model.coef_, "Intercept:", poly_model.intercept_)


### OUTPUT:
```
<img width="1920" height="1200" alt="Screenshot (171)" src="https://github.com/user-attachments/assets/b1c607f3-ca20-46bd-a001-6acebda655ff" />
<img width="1920" height="1200" alt="Screenshot (172)" src="https://github.com/user-attachments/assets/deedca1c-e718-4597-8334-974c597a8bb5" />




---

### RESULT:

Thus, the Python program for Linear and Polynomial Trend Estimation has been executed successfully.
