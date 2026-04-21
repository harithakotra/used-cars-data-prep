import pandas as pd
# Load dataset
url = "https://drive.google.com/file/d/14Yxdyp_vTM94nBjNH2igpsfjvMgbefWO/view?pli=1"
df = pd.read_csv(url, skiprows=2)
# Basic inspection
print(f"Shape before cleaning: {df.shape}")
print("\nInfo:\n")
print(df.info())
print("\nMissing values (%):\n")
print(df.isnull().sum() * 100 / len(df))


# Cleaning Data
# Drop rows where selling_price is missing
df = df.dropna(subset=['selling_price'])
# Clean mileage, engine, max_power
cols_to_clean = ['mileage', 'engine', 'max_power']
for col in cols_to_clean:
    df[col] = df[col].astype(str).str.extract('(\d+\.?\d*)')  # extract numeric part
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())
# Remove unrealistic prices
df = df[(df['selling_price'] != 999999999) & (df['selling_price'] >= 10000)]
# Drop duplicates
df = df.drop_duplicates()
# Final shape
print(f"\nShape after cleaning: {df.shape} ")

# Encode Categorical Features
# Label encoding
# Transmission encoding
df['transmission_type'] = df['transmission_type'].map({
    'Manual': 0,
    'Automatic': 1
})
# One hot encoding
df = pd.get_dummies(df, columns=['fuel_type', 'seller_type'], drop_first=True)

# Print final columns
print("\nFinal columns:\n")
print(df.columns.tolist())


# Split and Compute Baseline MAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Define X and y
X = df.drop(columns=['selling_price'])
y = df['selling_price']
# Keep only numeric columns
X = X.select_dtypes(include=['int64', 'float64'])
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Baseline prediction (mean of y_train)
baseline_pred = [y_train.mean()] * len(y_test)
# Compute MAE
mae = mean_absolute_error(y_test, baseline_pred)
print(f"Baseline MAE: ₹{round(mae)}")