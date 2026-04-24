# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula.
   <img width="462" height="199" alt="image" src="https://github.com/user-attachments/assets/23c3791d-06e3-48b0-b5a7-7b8c595f2848" />
4. Compute the y -intercept of the line by using the formula:
   <img width="295" height="79" alt="image" src="https://github.com/user-attachments/assets/408686d1-eb89-45b6-8d43-043443c75bce" />
5. Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Syed Shamsheer Ali
RegisterNumber: 21225220114
*/
```# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# Step 1: Create the dataset
# ------------------------------
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored':  [35, 40, 50, 55, 60, 65, 70, 75, 80, 85]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

# ------------------------------
# Step 2: Split into X and Y
# ------------------------------
X = df[['Hours_Studied']]   # Feature (2D)
y = df['Marks_Scored']      # Target (1D)

# ------------------------------
# Step 3: Split data for training & testing
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 4: Create and train the model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# Step 6: Evaluate the model
# ------------------------------
print("\nModel Evaluation:")
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ------------------------------
# Step 7: Visualize results
# ------------------------------
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Simple Linear Regression: Hours vs Marks')
plt.legend()
plt.show()

# ------------------------------
# Step 8: Predict for new data
# ------------------------------
hours = float(input("\nEnter number of study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for studying {hours} hours = {predicted_marks[0]:.2f}")
```

## Output:
<img width="937" height="771" alt="ML EXP 2" src="https://github.com/user-attachments/assets/f3803168-785a-4b73-9ce3-778072d1bfe1" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
