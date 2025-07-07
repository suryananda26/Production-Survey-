# Production-Survey-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


# Step 1: Load the dataset
df=pd.read_csv('Production_survey.csv')
df.columns = df.columns.str.strip()

# Step 4: Display a few rows and column info
print("Dataset Preview:")
print(df.head())
print("\nColumns:", df.columns.tolist())
# Step 5: Use Existing Annual Turnover Column
if 'Annual turnover in rupees' in df.columns:
    df['Annual_Turnover'] = df['Annual turnover in rupees']
else:
    raise ValueError("Column 'Annual turnover in rupees' not found in the dataset.")

print("\nAnnual turnover loaded from 'Annual turnover in rupees' column.")


# Ensure 'Company Name' and 'Annual_Turnover' are available
if 'Name of the Industry' in df.columns and 'Annual_Turnover' in df.columns:
    # Sort the DataFrame by Annual_Turnover in descending order
    top_company = df.sort_values(by='Annual_Turnover', ascending=False).iloc[0]

    print("🏆 Company with the Highest Annual Turnover:")
    print(f"Name of the Industry : {top_company['Name of the Industry']}")
    print(f"Annual Turnover (INR): ₹{top_company['Annual_Turnover']:,.2f}")
    if 'Address' in df.columns:
        print(f"Location     : {top_company['Address']}")
else:
    print("Required columns 'Company Name' or 'Annual_Turnover' not found.")

from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

# Recreate train-test split with indices preserved
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Also grab corresponding rows from the original DataFrame (for company names)
df_test = df.loc[X_test.index]

models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n🔍 {name} Performance:")
    print(f"Accuracy (R² Score): {r2 * 100:.2f}%")
    print(f"MSE: {mse:,.2f}")

    # Display top 5 predictions with company names
    result_df = pd.DataFrame({
        'Name of the Industry': df_test['Name of the Industry'].values,
        'Actual Annual Turnover': y_test.values,
        'Predicted Annual Turnover': y_pred
    })

    print("\nTop 5 Predicted vs Actual Annual Turnovers (with Company Name):")
    print(result_df.head(5))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

d = pd.read_csv('Production_survey.csv')

# 2. Preprocessing (Assuming 'Annual_Turnover' is your target and other columns are features)
#    a. Select features (X) and target (y)
X = d.drop(columns=['Annual_Turnover'])  # Features (all columns except 'Annual_Turnover')
y = d['Annual_Turnover']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n🔍 {name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Custom Accuracy (1 - |R²|): {(1 - abs(r2)) * 100:.2f}%")
    print(f"MSE: {mse:,.2f}")
if 'IP Address' in df.columns and 'Branch/Place' in df.columns:
    # Convert turnover to numeric (safety check)
    df['Annual_Turnover'] = pd.to_numeric(df['Annual_Turnover'], errors='coerce')

    # Create a new column to combine Branch and IP
    df['Location_Label'] = df['Branch/Place'].astype(str) + " (" + df['IP Address'].astype(str) + ")"

    # Group by this combined label
    location_avg = df.groupby('Location_Label')['Annual_Turnover'].mean().sort_values(ascending=False)

    # Show top 5
    print("\nTop 3 Best Locations to Install a New Company (by Avg Annual Turnover):")
    print(location_avg.head(3))

    # Plotting
    import seaborn as sns
    plt.figure(figsize=(10, 5))
    sns.barplot(x=location_avg.head(3).values, y=location_avg.head(3).index, palette="viridis")
    plt.xlabel("Average Annual Turnover")
    plt.title("Top 3 Locations to Start a New Company")
    plt.tight_layout()
    plt.show()

else:
    print("Required columns 'IP Address' and/or 'Branch' not found.")
import pandas as pd
inventory_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['stock','inventory'])]

# Check if we found any
if not inventory_cols:
    raise ValueError("No inventory survey columns found with the expected question pattern.")

# Calculate Inventory Awareness Score (mean of repeated survey answers)
for col in inventory_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' converts non-numeric to NaN

df['Inventory Awareness Score'] = df[inventory_cols].mean(axis=1, numeric_only=True)

#  Group by Company and calculate the average score
inventory_scores = df.groupby('Name of the Industry')['Inventory Awareness Score'].mean().sort_values()

#  Display Top 10 Companies with Best Inventory Practices
print("🏆 Top 10 Companies with Strong Inventory Management Awareness (Lower = Better):")
print(inventory_scores.head(10))

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(x=inventory_scores.head(10).values, y=inventory_scores.head(10).index, palette="crest")
plt.title("Top 10 Companies with Best Inventory Awareness")
plt.xlabel("Average Inventory Awareness Score (1=Strongly Agree, 5=Strongly Disagree)")
plt.tight_layout()
plt.show()

# Drop non-informative or identifier columns
columns_to_exclude = ['Email Address', 'First Name', 'Last Name', 'Custom Data 1', 'Unnamed: 22']
df_clean = df.drop(columns=columns_to_exclude, errors='ignore')

# Keep only numeric columns
numeric_cols = df_clean.select_dtypes(include='number').columns
correlation_matrix = df_clean[numeric_cols].corr()

# Focus on correlation with Annual_Turnover
target_corr = correlation_matrix['Annual_Turnover'].drop('Annual_Turnover')

# Sort by strongest absolute correlation
strongest_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)

# Display clean and ranked output
print("\n🔍 Features Most Correlated with Annual Turnover:\n")
for feature, corr in strongest_corr.items():
    if not pd.isna(corr):
        label = "🔺" if corr > 0 else "🔻"
        print(f"{feature:<30} {label} Correlation: {corr:.2f}")
#Distribution of Annual Turnover:
plt.figure(figsize=(8, 6))
sns.histplot(df['Annual_Turnover'], kde=True, bins=30)
plt.title('Distribution of Annual Turnover')
plt.xlabel('Annual Turnover')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate average turnover per industry
top_industries = df.groupby('Name of the Industry')['Annual_Turnover'].mean().sort_values(ascending=False).head(10).index

# Filter the dataset
filtered_df = df[df['Name of the Industry'].isin(top_industries)]

# Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Name of the Industry', y='Annual_Turnover', data=filtered_df, palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.title('Annual Turnover Distribution - Top 10 Industries by Avg Turnover')
plt.xlabel('Industry')
plt.ylabel('Annual Turnover')
plt.tight_layout()
plt.show()
import pandas as pd
df = pd.read_csv('Production_survey.csv', delimiter=',', header=0)
print(df.columns.tolist())

import pandas as pd

# Load the data
df = pd.read_csv('Production_survey.csv', header=0)
product_columns = [
    'Fruit and vegetables', 'Sweets', 'Grain processing', 'Spices',
    'Milk', 'Cocoa products', 'Beverages', 'Soya-based',
    'Alcoholic Beverages', 'Meat', 'Mineral water', 'Fisheries',
    'Other (please specify)'
]

# Get the current column names
current_columns = df.columns.tolist()

# Rename the product-related columns
new_columns = current_columns[:13] + product_columns + current_columns[26:]

# Ensure the number of columns matches
if len(new_columns) == len(current_columns):
    df.columns = new_columns
else:
    print(f"Warning: Column length mismatch. Expected {len(new_columns)}, got {len(current_columns)}")
    print("Current columns:", current_columns)
    print("Attempted new columns:", new_columns)
    raise ValueError("Column length mismatch")

# Print the cleaned column names
print(df.columns.tolist())

def get_products_by_industry_with_input(df):
    possible_industry_columns = ['Name of the Industry', 'Name of the industry', 'Name of Industry']
    industry_column = None

    # Check available columns
    print("Available columns:", df.columns.tolist())

    # Look for a matching industry column
    for col in possible_industry_columns:
        if col in df.columns:
            industry_column = col
            break

    if industry_column is None:
        # If not found, try a partial match or suggest the closest column
        for col in df.columns:
            if 'Industry' in col:
                industry_column = col
                print(f"Using '{industry_column}' as the industry column (best guess)")
                break

    if industry_column is None:
        print("Error: No industry column found in the DataFrame")
        return

    # Define product columns
    product_columns = [
        'Fruit and vegetables', 'Sweets', 'Grain processing', 'Spices',
        'Milk', 'Cocoa products', 'Beverages', 'Soya-based',
        'Alcoholic Beverages', 'Meat', 'Mineral water', 'Fisheries',
        'Other (please specify)'
    ]

    while True:
        # Get user input
        industry_name = input("Enter the name of the industry (or 'quit' to exit): ")

        # Check for exit condition
        if industry_name.lower() == 'quit':
            print("Exiting...")
            break

        # Find matching industry (case-insensitive)
        industry_data = df[df[industry_column].str.lower() == industry_name.lower()]

        if industry_data.empty:
            print(f"No industry found with the name '{industry_name}'")
        else:
            # Get the first matching row (assuming unique industry names)
            industry_row = industry_data.iloc[0]

            # Find products that have values
            products = []
            for col in product_columns:
                if col in df.columns and pd.notna(industry_row[col]) and industry_row[col] != '':
                    products.append(col)

            if not products:
                print(f"'{industry_name}' does not specify any products")
            else:
                print(f"Products produced by {industry_name}: {', '.join(products)}")

        print()  # Add a blank line for readability

# Run with your DataFrame
get_products_by_industry_with_input(df)
