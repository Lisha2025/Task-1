import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 1. Load the Iris dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("✅ Loaded dataset:")
    print(df.head())  # Display first 5 rows
    print("\n📊 Columns:", df.columns.tolist())
    print("\n📈 Dataset shape:", df.shape)
    return df

# 2. Define preprocessing pipeline
def build_preprocessing_pipeline(df):
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        print("\nℹ️ 'Id' column dropped")

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    print("\n🔢 Numeric features:", numeric_features)
    print("🔠 Categorical features:", categorical_features)

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    print("\n🔧 Preprocessing pipeline created")
    return preprocessor, numeric_features, categorical_features

# 3. Preprocess and split
def preprocess_and_split(df, target_column):
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("\n📌 Target column:", target_column)
    print("📊 Features shape before processing:", X.shape)

    preprocessor, _, _ = build_preprocessing_pipeline(X)
    X_processed = preprocessor.fit_transform(X)

    print("✅ Features shape after preprocessing:", X_processed.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    print("\n🧪 X_train shape:", X_train.shape)
    print("🧪 X_test shape:", X_test.shape)
    print("🎯 y_train shape:", y_train.shape)
    print("🎯 y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test

# 4. Run everything
if __name__ == "__main__":
    df = load_data("Iris.csv")
    X_train, X_test, y_train, y_test = preprocess_and_split(df, target_column="Species")
    print("\n✅✅ Data preprocessing pipeline completed successfully.")
