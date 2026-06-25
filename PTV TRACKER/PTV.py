import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv('data/ptv_delays.csv')

# Drop nulls
df = df.dropna(subset=['is_delayed'])

# Features
features = ['hour', 'day_of_week', 'temperature', 'windspeed', 'precipitation']

# Encode route_id and stop_id as categories
df['route_encoded'] = df['route_id'].astype('category').cat.codes
df['stop_encoded'] = df['stop_id'].astype('category').cat.codes
features += ['route_encoded', 'stop_encoded']

X = df[features]
y = df['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBClassifier
import numpy as np

model = XGBClassifier(
    scale_pos_weight=5,  # ratio of non-delayed to delayed
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Use 0.3 threshold instead of default 0.5
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.3).astype(int)

print(classification_report(y_test, preds))
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved!")


import matplotlib.pyplot as plt
import numpy as np

feature_names = ['hour', 'day_of_week', 'temperature', 'windspeed', 'precipitation', 'route_encoded', 'stop_encoded']

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[indices])
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()