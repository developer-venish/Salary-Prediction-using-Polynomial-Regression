# Salary-Prediction-using-Polynomial-Regression
ML Python Project

Linear Regressiom

![](https://github.com/developer-venish/Salary-Prediction-using-Polynomial-Regression/blob/main/graph1.png)

---------------------------------------------------------------------------------------

Polynomial Regression

![](https://github.com/developer-venish/Salary-Prediction-using-Polynomial-Regression/blob/main/graph1.png)

---------------------------------------------------------------------------------------

Prediction

![](https://github.com/developer-venish/Salary-Prediction-using-Polynomial-Regression/blob/main/prediction.png)

---------------------------------------------------------------------------------------

Note :- All the code in this project has been tested and run successfully in Google Colab. I encourage you to try running it in Colab for the best experience and to ensure smooth execution. Happy coding!

---------------------------------------------------------------------------------------

The code provided demonstrates the use of linear regression and polynomial regression for predicting salary based on the level of a person. Here's an explanation of the code:

1. Import necessary libraries:
   - `pandas` for data manipulation and analysis.
   - `files` module from `google.colab` for uploading files in Google Colab.
   - `matplotlib.pyplot` for data visualization.

2. Use `files.upload()` to upload the dataset file. Make sure the file is named `dataset.csv`.

3. Read the dataset using `pd.read_csv('dataset.csv')` and assign it to the `dataset` variable.

4. Print the shape of the dataset using `dataset.shape` to get the number of rows and columns.
   - This helps to understand the size of the dataset.

5. Print the first 5 rows of the dataset using `dataset.head(5)`.
   - This allows you to inspect the structure and content of the dataset.

6. Extract the independent variable (X) from the dataset by selecting all columns except the last one using `dataset.iloc[:, :-1].values`. This creates a NumPy array of X values.

7. Extract the dependent variable (Y) from the dataset by selecting only the last column using `dataset.iloc[:, -1].values`. This creates a NumPy array of Y values.

8. Create an instance of `LinearRegression()` called `modelLR`.

9. Fit the linear regression model using `modelLR.fit(X, Y)`. This trains the model on the given X and Y data.

10. Visualize the linear regression results:
    - Create a scatter plot using `plt.scatter(X, Y, color="red")` to show the actual data points.
    - Plot the regression line using `plt.plot(X, modelLR.predict(X))` to display the predicted values.
    - Add a title to the plot using `plt.title("Linear Regression")`.
    - Label the x-axis using `plt.xlabel("Level")`.
    - Label the y-axis using `plt.ylabel("Salary")`.
    - Show the plot using `plt.show()`.

11. Create an instance of `PolynomialFeatures` called `modelPR` with a degree of 2.
    - This will transform the original features (X) into polynomial features.

12. Transform the original features (X) into polynomial features using `modelPR.fit_transform(X)` and assign it to `xPoly`.

13. Create another instance of `LinearRegression()` called `modelPLR`.

14. Fit the polynomial regression model using `modelPLR.fit(xPoly, Y)`.

15. Visualize the polynomial regression results:
    - Create a scatter plot using `plt.scatter(X, Y, color="red")` to show the actual data points.
    - Plot the predicted values using `plt.plot(X, modelPLR.predict(modelPR.fit_transform(X)))`.
    - Add a title to the plot using `plt.title("Polynomial Regression")`.
    - Label the x-axis using `plt.xlabel("Level")`.
    - Label the y-axis using `plt.ylabel("Salary")`.
    - Show the plot using `plt.show()`.

16. Predict the salary for a given level (x = 5) using `modelPLR.predict(modelPR.fit_transform([[x]]))`.
    - This demonstrates how to use the trained polynomial regression model to make predictions.
    - The predicted salary is stored in the `salaryPred` variable.

17. Print the predicted salary using `print('Salary of person with Level{0} is {1}'.format(x, salaryPred))`.

Overall, the code performs linear regression and polynomial regression on the given dataset and visualizes the results using scatter plots and regression lines. It also demonstrates how to make predictions using the trained polynomial regression model.

---------------------------------------------------------------------------------------

Polynomial regression is a form of regression analysis in which the relationship between the independent variable(s) and dependent variable is modeled as an nth-degree polynomial. It extends the concept of linear regression by allowing for nonlinear relationships between the variables.

In polynomial regression, the relationship between the independent variable (X) and dependent variable (Y) is approximated using a polynomial function of the form:

Y = b0 + b1*X + b2*X^2 + b3*X^3 + ... + bn*X^n

Here, X represents the independent variable, Y represents the dependent variable, and the coefficients b0, b1, b2, ..., bn are the parameters to be estimated.

Polynomial regression can capture more complex relationships between variables compared to linear regression. By introducing higher-degree polynomial terms, it can account for curvature and nonlinear patterns in the data. This flexibility allows polynomial regression to fit the data more closely and potentially provide better predictions.

The degree of the polynomial determines the complexity of the model. Higher degrees can lead to overfitting if the model becomes too complex for the given data, while lower degrees may not capture all the important patterns in the data. Therefore, the degree of the polynomial needs to be carefully chosen based on the specific problem and the characteristics of the data.

Polynomial regression can be applied when there is a suspicion that the relationship between variables is nonlinear or when the linear regression model does not adequately fit the data. It is commonly used in various fields such as economics, physics, social sciences, and engineering to analyze and model nonlinear relationships between variables.

---------------------------------------------------------------------------------------
