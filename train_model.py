import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("archive/yield_df.csv")

maize_df = df[df['Item'] == 'Maize']

maize_df = maize_df.drop(columns=['Unnamed: 0', 'Item'])

le = LabelEncoder()
maize_df['Area'] = le.fit_transform(maize_df['Area'])

X = maize_df.drop('hg/ha_yield', axis=1)
y = maize_df['hg/ha_yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

param_grid = {
    'max_depth':[2,3,4,5,6,8,10,12,15]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

joblib.dump(grid.best_estimator_, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model Saved Successfully!")