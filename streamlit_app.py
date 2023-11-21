import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Load your dataset
df = pd.read_csv("ncbirths.csv")

# Title and description
st.title("Male Babys' Birthday Weight Analysis with Streamlit")
st.write("In 2004, the state of North Carolina released a large dataset containing information on births recorded in the state.")
st.write("In this dataset it has 1000 observations on the following 13 variables.")
st.write("1. fage: Father's age in years.")
st.write("2. mage: Mother's age in years.")
st.write("3. mature: Maturity status of mother.")
st.write("4. weeks: Length of pregnancy in weeks.")
st.write("5. premie: Whether the birth was classified as premature (premie) or full-term.")
st.write("6. visits: Number of hospital visits during pregnancy.")
st.write("7. gained: Weight gained by mother during pregnancy in pounds.")
st.write("8. weight: Weight of the baby at birth in pounds.")
st.write("9. lowbirthweight: Whether the baby was classified as low birthweight (low) or not (not low).")
st.write("10. gender: Gender of the baby, female or male.")
st.write("11. habit: Status of the mother as a nonsmoker or a smoker.")
st.write("12. marital: Whether the mother is married or not married at birth.")
st.write("13. whitemom: Whether the mom is white or not white.")
st.write("We are gonna to perform an analysis on the factors related to a baby's birth weight, specifically focusing on the maternal factors.")
st.write("Following the 'ceteris paribus' rule, we have exclusively chosen data related to male infants born to white mothers and focused our study on the factors originating from the mother that may affect the baby's birth weight.")

#have some rough ideas about the dataframe
with st.expander("Click to see the top 20 rows of the data frame"):
    # Display the top 20 rows of the DataFrame
    st.dataframe(df.head(20))


# Data preprocessing
st.sidebar.header("Data Preprocessing")
st.write("Now, let's preprocess the data. Firstly, we will filter out data where the gender is 'male', and the mom's race is 'white'. Then, we will drop the father-related variable column: 'fage'. Since prematurity is defined by the length of pregnancy in weeks, we will only keep 'weeks' and drop 'premie'. Maturity status is decided by the mom's age, so we will drop 'mature'. We choose 'weight' as our dependent varible and drop 'lowbirthweight'.")

# Filter data by white mothers and male babies
filtered_df = df[(df["whitemom"] == "white") & (df["gender"] == "male")]

# Handle missing values by filling them with the mean
filtered_df.fillna(filtered_df.mean(), inplace=True)

# Drop unnecessary columns
filtered_df.drop(columns=["fage", "marital", "mature", "premie", "lowbirthweight"], inplace=True)

# Histograms
st.subheader("Histograms")
numeric_columns = filtered_df.select_dtypes(include=["number"]).columns

# Calculate the number of rows and columns for the grid
num_rows = len(numeric_columns) // 3 + (len(numeric_columns) % 3 > 0)
num_cols = min(len(numeric_columns), 3)

# Create a figure with subplots and adjust hspace for spacing
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4 * num_rows), 
                         gridspec_kw={'hspace': 0.5})

# Iterate through numeric columns and create histograms
for i, column in enumerate(numeric_columns):
    row_idx = i // 3
    col_idx = i % 3
    ax = axes[row_idx, col_idx]
    
    sns.histplot(data=filtered_df, x=column, bins=20, kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

# Remove empty subplots
for i in range(len(numeric_columns), num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

# Adjust spacing between subplots
plt.tight_layout()

# Display the figure using st.pyplot
st.pyplot(fig)
st.write("From the histograms we can see...")

# Sidebar for data exploration
st.sidebar.header("Data Exploration")

st.subheader("Bivariate Plots")
# Create subplots with a 2-row, three-column grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Scatter plot: Weight vs. Mage
axes[0, 0].scatter(filtered_df['mage'], filtered_df['weight'], alpha=0.5)
axes[0, 0].set_xlabel('Mage')
axes[0, 0].set_ylabel('Weight')
axes[0, 0].set_title('Weight vs. Mage')

# Box plot: Weight vs. Habit (Smoking Status) without gridlines
filtered_df.boxplot(column='weight', by='habit', ax=axes[0, 1], grid=False)
axes[0, 1].set_xlabel('Habit (Smoking Status)')
axes[0, 1].set_ylabel('Weight')
axes[0, 1].set_title('Weight vs. Habit (Smoking Status)')

# Scatter plot: Weight vs. Weeks
axes[0, 2].scatter(filtered_df['weeks'], filtered_df['weight'], alpha=0.5)
axes[0, 2].set_xlabel('Weeks')
axes[0, 2].set_ylabel('Weight')
axes[0, 2].set_title('Weight vs. Weeks')

# Scatter plot: Weight vs. Visits
axes[1, 0].scatter(filtered_df['visits'], filtered_df['weight'], alpha=0.5)
axes[1, 0].set_xlabel('Visits')
axes[1, 0].set_ylabel('Weight')
axes[1, 0].set_title('Weight vs. Visits')

# Scatter plot: Weight vs. Gained
axes[1, 1].scatter(filtered_df['gained'], filtered_df['weight'], alpha=0.5)
axes[1, 1].set_xlabel('Gained')
axes[1, 1].set_ylabel('Weight')
axes[1, 1].set_title('Weight vs. Gained')

# Remove empty subplot
fig.delaxes(axes[1, 2])

# Adjust spacing between subplots
plt.tight_layout()

# Display the figure using st.pyplot
st.pyplot(fig)



# Correlation matrix
st.subheader("Correlation Matrix")
# Calculate the correlation matrix
correlation_matrix = filtered_df.corr()
# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
st.pyplot(plt)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Linear regression analysis with statsmodels
st.sidebar.header("Linear Regression Analysis")
st.write("Now, let's perform linear regression using statsmodels.")

# One-hot encoding for habit
filtered_df_encoded = pd.get_dummies(filtered_df, columns=["habit"], drop_first=True)

# Define independent and dependent variables
X = filtered_df_encoded[["mage", "weeks", "visits", "habit_smoker"]]
y = filtered_df_encoded["weight"]

# Convert columns to appropriate data types (float)
X["mage"] = X["mage"].astype(float)
X["weeks"] = X["weeks"].astype(float)
X["visits"] = X["visits"].astype(float)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Add a constant term (intercept) to the independent variables
X = sm.add_constant(X)

# Create a linear regression model with statsmodels
model_stats = sm.OLS(y, X)

# Fit the model
results = model_stats.fit()

# Get the summary of the regression results
st.subheader("Regression Results")
st.text(results.summary())

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
st.header("Linear Regression Analysis (Using scikit-learn)")
st.write("Here are the results of linear regression using statsmodels:")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R2): {r2}")


# Define independent and dependent variables
X = filtered_df_encoded[["weeks", "habit_smoker"]]
y = filtered_df_encoded["weight"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Add a constant term (intercept) to the independent variables
X = sm.add_constant(X)

# Create a linear regression model with statsmodels
model_stats = sm.OLS(y, X)



# Fit the model
results = model_stats.fit()

# Get the summary of the regression results

st.text(results.summary())

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results

st.write("Here are the results of linear regression using statsmodels:")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R2): {r2}")






