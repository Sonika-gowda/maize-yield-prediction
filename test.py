import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

# STEP 1: LOAD DATASET

df = pd.read_csv("archive/yield_df.csv")

print("Original Dataset Shape:", df.shape)
# STEP 2: FILTER ONLY MAIZE

maize_df = df[df['Item'] == 'Maize']

print("Maize Dataset Shape:", maize_df.shape)

# STEP 3: REMOVE UNUSED COLUMNS

maize_df = maize_df.drop(columns=['Unnamed: 0', 'Item'])

# STEP 4: ENCODE AREA

le = LabelEncoder()
maize_df['Area'] = le.fit_transform(maize_df['Area'])

# STEP 5: FEATURES AND TARGET

X = maize_df.drop('hg/ha_yield', axis=1)

y = maize_df['hg/ha_yield']

print("\nFeature Columns:")
print(X.columns.tolist())

# STEP 6: TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 8, 10, 12, 15]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Max Depth:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

# STEP 9: EVALUATION

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")

print("MAE :", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("R2 Score:", round(r2, 4))