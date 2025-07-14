# Salary Prediction & Hypothesis Testing: Is Education the Key to Higher Pay?

This project investigates the hypothesis:  
> **"Education is a significant factor in determining a person's salary."**

Using real-world job data and machine learning, we build two regression models — one with education as a feature, and one without — and compare their predictive performance to assess the importance of education.

---

## Dataset Overview

**File:** `Salary Data.csv`

### Target:
- `Salary` (numerical)

### Features:

| Feature               | Type         | Description                                |
|-----------------------|--------------|--------------------------------------------|
| Age                   | Numerical    | Age of the individual                      |
| Gender                | Categorical  | Male / Female                              |
| Education Level       | Categorical  | Bachelor's, Master's, PhD                  |
| Job Title             | Categorical  | Over 100 unique titles                     |
| Years of Experience   | Numerical    | Float, e.g. 3.0, 8.5                        |

---

## Machine Learning Approach

We use **multiple linear regression** with `scikit-learn` pipelines.

### Model 1: With Education
- Uses all 5 features.
- OneHotEncoder for categorical vars.
- Trained/tested with `train_test_split`.

### Model 2: Without Education
- Drops `Education Level` to evaluate its impact.
- Same encoding & model process used.

---

## Evaluation Metrics

| Metric         | With Education | Without Education |
|----------------|----------------|-------------------|
| R² Score       | Higher         | Lower             |
| MAE            | Lower          | Higher            |

Higher R² and lower MAE in Model 1 support the **hypothesis that education impacts salary**.

---

## Feature Importance

- Coefficients from the regression model are extracted and ranked.
- Visualized as bar charts.
- `Education Level_Master's`, `Education Level_PhD` appear among the top features.

---

## Future Perspective User Interaction

The script supports **interactive salary prediction**:

```text
Enter user details to predict salary:
Age: 30
Gender (Male/Female): Male
Education Level (Bachelor's/Master's/PhD): Master's
Job Title (e.g., Software Engineer): Data Analyst
Years of Experience: 6

--- Salary Predictions ---
Predicted Salary (WITH Education): ₹95,400.22
Predicted Salary (WITHOUT Education): ₹83,732.18
