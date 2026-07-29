import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

#tests best threshold to use, this is important since there is a class imbalance since if we used the base
#threshold =0.5 the model would rarely predict a high probabilty for delays because of the proportional difference in data
# so almost always predict non_delayed

def best_threshold(y_true, probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = f1s[:-1].argmax()
    return thresholds[best_idx], f1s[best_idx]

df = pd.read_csv('data/ptv_delays.csv', low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'])

#=======================================================================================================================
#data analysis
#=======================================================================================================================

print("=" * 50)
print("BASIC SHAPE")
print("=" * 50)
print(f"Rows: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Days covered: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print(f"Unique routes: {df['route_id'].nunique()}")
print(f"Unique stops: {df['stop_id'].nunique()}")

print("\n" + "=" * 50)
print("CLASS BALANCE (is_delayed)")
print("=" * 50)
print(df['is_delayed'].value_counts())
print(df['is_delayed'].value_counts(normalize=True) * 100)

print("\n" + "=" * 50)
print("MISSING DATA")
print("=" * 50)
print(df.isnull().sum())

plt.figure(figsize=(10, 5))
hourly = df.groupby('hour')['is_delayed'].mean() * 100
hourly.plot(kind='bar', color='steelblue')
plt.title('Delay Rate by Hour of Day (Melbourne time)')
plt.ylabel('% of trips delayed')
plt.xlabel('Hour')
plt.tight_layout()
plt.savefig('eda_delay_by_hour.png')
plt.show()

plt.figure(figsize=(8, 5))
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily = df.groupby('day_of_week')['is_delayed'].mean() * 100
daily.index = day_names
daily.plot(kind='bar', color='coral')
plt.title('Delay Rate by Day of Week')
plt.ylabel('% of trips delayed')
plt.tight_layout()
plt.savefig('eda_delay_by_day.png')
plt.show()

plt.figure(figsize=(10, 6))
route_delay = df.groupby('route_id')['is_delayed'].mean().sort_values(ascending=False).head(15) * 100
route_delay.plot(kind='barh', color='indianred')
plt.title('Top 15 Most-Delayed Routes')
plt.xlabel('% of trips delayed')
plt.tight_layout()
plt.savefig('eda_worst_routes.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df['precipitation'], df['delay_minutes'], alpha=0.02)
plt.title('Delay Minutes vs Precipitation')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Delay (minutes)')
plt.tight_layout()
plt.savefig('eda_weather_delay.png')
plt.show()

plt.figure(figsize=(10, 5))
df[df['delay_minutes'] > 0]['delay_minutes'].clip(upper=30).hist(bins=50)
plt.title('Distribution of Delay Lengths (delays only, clipped at 30 min)')
plt.xlabel('Delay (minutes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda_delay_distribution.png')
plt.show()

plt.figure(figsize=(8, 6))
numeric_cols = ['hour', 'day_of_week', 'temperature', 'windspeed', 'precipitation', 'is_delayed']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('eda_correlation.png')
plt.show()

#=======================================================================================================================
#pre processing
#=======================================================================================================================

df = df.dropna(subset=['is_delayed'])
df = df[(df['delay_minutes'] > -30) & (df['delay_minutes'] < 120)]

#this is always active becauses there always atleast one alert active on metro so this column is useless
df = df.drop(columns=['has_network_alert'])

#for the missing columns we saw in the data analysis we fill them up with the median values
for col in ['temperature', 'windspeed', 'precipitation', 'weather_code']:
    df[col] = df[col].fillna(df[col].median())

#creates a feature which tells us if the delays happened during a peak hour on a weekday
df['is_peak_hour'] = (
    ((df['hour'].between(7, 9)) | (df['hour'].between(16, 18)))
    & (df['day_of_week'] < 5)
).astype(int)

df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

#converts the routes and stops as categories and encodes them so they stay consistent even when different
#data is used

route_categories = df['route_id'].astype('category').cat.categories
stop_categories = df['stop_id'].astype('category').cat.categories

df['route_encoded'] = pd.Categorical(df['route_id'], categories=route_categories).codes
df['stop_encoded'] = pd.Categorical(df['stop_id'], categories=stop_categories).codes

with open('encoders.pkl', 'wb') as f:
    pickle.dump({'route_categories': route_categories, 'stop_categories': stop_categories}, f)



df = df.sort_values(['route_id', 'timestamp']).reset_index(drop=True)

#groups the same trip into one row so we don't treat repeats of the stop as the same thing
#averages those delays in that trip to save in one row

trip_level = (
    df.groupby(['route_id', 'trip_id', 'timestamp'])['delay_minutes']
    .mean()
    .reset_index()
    .sort_values(['route_id', 'timestamp'])
)


#creating a new feature that lets the a trip see the delay over the past hour, also shifted it to make sure that
# there isnt any data leaks and it can't see its own delay
trip_level = trip_level.set_index('timestamp')
trip_level['recent_route_delay'] = (
    trip_level.groupby('route_id')['delay_minutes']
    .apply(lambda x: x.shift(1).rolling('60min').mean())
    .reset_index(level=0, drop=True)
)
trip_level = trip_level.reset_index()

#merge this feature onto the dataframe
df = df.merge(
    trip_level[['route_id', 'trip_id', 'timestamp', 'recent_route_delay']],
    on=['route_id', 'trip_id', 'timestamp'],
    how='left'
)

#since we shifted it the first rows in delays would be empty so we fill those with zero instead of null
df['recent_route_delay'] = df['recent_route_delay'].fillna(0)

df = df.sort_values('timestamp').reset_index(drop=True)

features = ['hour', 'day_of_week', 'temperature', 'windspeed', 'precipitation',
            'active_alerts', 'is_peak_hour', 'is_weekend', 'route_encoded', 'stop_encoded',
            'recent_route_delay']

#=======================================================================================================================
#Hyperparamatising
#=======================================================================================================================
X = df[features]
y = df['is_delayed']

from sklearn.model_selection import TimeSeriesSplit


# creates 5 splits in chronological order in out data
tscv = TimeSeriesSplit(n_splits=5)

#splits the data into training and test sets
all_splits = list(tscv.split(X))
X_trainfull, X_test = X.iloc[all_splits[-1][0]], X.iloc[all_splits[-1][1]]
y_trainfull, y_test = y.iloc[all_splits[-1][0]], y.iloc[all_splits[-1][1]]


#create validation set out of training data so we don't leak any of our test data while we hyperparamaterise
val_split_idx = int(len(X_trainfull) * 0.85)
X_train, X_val = X_trainfull.iloc[:val_split_idx], X_trainfull.iloc[val_split_idx:]
y_train, y_val = y_trainfull.iloc[:val_split_idx], y_trainfull.iloc[val_split_idx:]

#gives us scale of the class imbalance between delayed vs non_delayed to help us parametise weight
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()


#testing weight values recording the threshold and f1 score, keep the best f1 score as the best
print("\n" + "=" * 50)
print("SCALE_POS_WEIGHT SEARCH (XGBoost, last fold)")
print("=" * 50)
best_weight = 1
best_weight_f1 = -1
for weight in [1, 3, 5, 10, 20, scale_pos]:
    temp_model = XGBClassifier(scale_pos_weight=weight, n_estimators=200, n_jobs=-1, random_state=42)
    temp_model.fit(X_train, y_train)
    probs = temp_model.predict_proba(X_val)[:, 1]
    threshold, f1 = best_threshold(y_val, probs)
    print(f"weight={weight}: best_f1={f1:.3f} at threshold={threshold:.2f}")
    if f1 > best_weight_f1:
        best_weight_f1 = f1
        best_weight = weight
print(f"-> best weight found: {best_weight} (f1={best_weight_f1:.3f})")

print("\n" + "=" * 50)
print("SCALE_POS_WEIGHT SEARCH (LightGBM, last fold)")
print("=" * 50)
best_lgbm_weight = 1
best_lgbm_weight_f1 = -1
for weight in [1, 3, 5, 10, 20, scale_pos]:
    temp_model = LGBMClassifier(scale_pos_weight=weight, n_estimators=200, n_jobs=-1, random_state=42, verbose=-1)
    temp_model.fit(X_train, y_train)
    probs = temp_model.predict_proba(X_val)[:, 1]
    threshold, f1 = best_threshold(y_val, probs)
    print(f"weight={weight}: best_f1={f1:.3f} at threshold={threshold:.2f}")
    if f1 > best_lgbm_weight_f1:
        best_lgbm_weight_f1 = f1
        best_lgbm_weight = weight
print(f"-> best weight found: {best_lgbm_weight} (f1={best_lgbm_weight_f1:.3f})")


#tests whether using none balanced of balanced class_weight is better
print("\n" + "=" * 50)
print("CLASS_WEIGHT SEARCH (LogisticRegression / RandomForest, last fold)")
print("=" * 50)
best_class_weight = {}
for name, builder in [
    ('LogisticRegression', lambda cw: LogisticRegression(max_iter=5000, class_weight=cw)),
    ('RandomForest', lambda cw: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight=cw)),
]:
    best_cw = None
    best_cw_f1 = -1
    for cw in [None, 'balanced']:
        temp_model = builder(cw)
        temp_model.fit(X_train, y_train)
        probs = temp_model.predict_proba(X_val)[:, 1]
        threshold, f1 = best_threshold(y_val, probs)
        print(f"{name} class_weight={cw}: best_f1={f1:.3f} at threshold={threshold:.2f}")
        if f1 > best_cw_f1:
            best_cw_f1 = f1
            best_cw = cw
    best_class_weight[name] = best_cw
    print(f"-> best class_weight for {name}: {best_cw} (f1={best_cw_f1:.3f})")

#=======================================================================================================================
# Cross validation of models
#=======================================================================================================================
#testing how reliable and consistesnt are models are

model_builders = {
    'LogisticRegression': lambda: LogisticRegression(max_iter=5000, class_weight=best_class_weight['LogisticRegression']),
    'RandomForest': lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight=best_class_weight['RandomForest']),
    'XGBoost': lambda: XGBClassifier(scale_pos_weight=best_weight, n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42),
    'LightGBM': lambda: LGBMClassifier(scale_pos_weight=best_lgbm_weight, n_estimators=200, n_jobs=-1, random_state=42, verbose=-1),
}

print("\n" + "=" * 50)
print("TIME SERIES CROSS-VALIDATION (5 folds)")
print("=" * 50)

cv_results = []

for name, build_model in model_builders.items():
    fold_scores = {'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

    #we repeat 5 times fore each model where the training data grows each run to see how consistent the models are
    #and how their performance is effected when they are given more data to train on
    for fold_num, (train_idx, test_idx) in enumerate(all_splits, start=1):
        # split the data into training and testing to compare how consistent the model is and how the amount of data given
        #to train affects f1 score
        X_tr_full, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr_full, y_te = y.iloc[train_idx], y.iloc[test_idx]

        #out of the training set take out a little bit for the validation set
        fold_val_idx = int(len(X_tr_full) * 0.85)
        X_tr, X_fold_val = X_tr_full.iloc[:fold_val_idx], X_tr_full.iloc[fold_val_idx:]
        y_tr, y_fold_val = y_tr_full.iloc[:fold_val_idx], y_tr_full.iloc[fold_val_idx:]

        model = build_model()
        model.fit(X_tr, y_tr)

        #validation set we created earlier is used to find what threshold to use
        val_probs = model.predict_proba(X_fold_val)[:, 1]
        threshold, _ = best_threshold(y_fold_val, val_probs)

        #we see how well we do for that fold
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= threshold).astype(int)

        fold_scores['precision'].append(precision_score(y_te, preds))
        fold_scores['recall'].append(recall_score(y_te, preds))
        fold_scores['f1'].append(f1_score(y_te, preds))
        fold_scores['roc_auc'].append(roc_auc_score(y_te, probs))

        print(f"{name} fold {fold_num}: f1={fold_scores['f1'][-1]:.3f}, roc_auc={fold_scores['roc_auc'][-1]:.3f}")

    cv_results.append({
        'model': name,
        'precision_mean': np.mean(fold_scores['precision']),
        'precision_std': np.std(fold_scores['precision']),
        'recall_mean': np.mean(fold_scores['recall']),
        'recall_std': np.std(fold_scores['recall']),
        'f1_mean': np.mean(fold_scores['f1']),
        'f1_std': np.std(fold_scores['f1']),
        'roc_auc_mean': np.mean(fold_scores['roc_auc']),
        'roc_auc_std': np.std(fold_scores['roc_auc']),
    })
    print("-" * 50)

cv_results_df = pd.DataFrame(cv_results).sort_values('f1_mean', ascending=False)
print("\nCROSS-VALIDATION SUMMARY (mean ± std across 5 folds):")
print(cv_results_df)
cv_results_df.to_csv('model_comparison_cv.csv', index=False)

results = []

#=======================================================================================================================
# final model testing
#=======================================================================================================================

#train each final model on train+validation combined (now that tuning decisions are locked in),
#find its threshold using validation only, then evaluate ONCE on the true untouched test set

#combines validation and training set together to train model on as much data as possible
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

for name, build_model in model_builders.items():
    print(f"Training final {name} (on train+validation)...")
    #training model
    model = build_model()
    model.fit(X_trainval, y_trainval)


    #we train model with a proportion of the training set and test on the validation set to find what threshold
    #to use for our final model
    val_model = build_model()
    val_model.fit(X_train, y_train)
    val_probs = val_model.predict_proba(X_val)[:, 1]
    threshold, _ = best_threshold(y_val, val_probs)

    #evaluate how our model did on the test set
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(f"{name} — threshold (chosen on validation): {threshold:.2f}")

    results.append({
        'model': name,
        'threshold': threshold,
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probs),
    })

    with open(f'model_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(classification_report(y_test, preds))
    print("-" * 50)

results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
print(results_df)
results_df.to_csv('model_comparison.csv', index=False)

best_name = results_df.iloc[0]['model']
best_threshold_value = results_df.iloc[0]['threshold']
best_model = model_builders[best_name]()
best_model.fit(X_trainval, y_trainval)

with open('model_final.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model_final_meta.pkl', 'wb') as f:
    pickle.dump({'features': features, 'threshold': best_threshold_value, 'model_name': best_name}, f)

print(f"\nBest model: {best_name} (saved as model_final.pkl)")

print("\n" + "=" * 50)
print("SINGLE SPLIT (last fold) vs CROSS-VALIDATION MEAN")
print("=" * 50)
comparison = results_df[['model', 'f1', 'roc_auc']].rename(
    columns={'f1': 'single_split_f1', 'roc_auc': 'single_split_roc_auc'}
).merge(
    cv_results_df[['model', 'f1_mean', 'f1_std', 'roc_auc_mean', 'roc_auc_std']],
    on='model'
)
print(comparison)
comparison.to_csv('single_split_vs_cv.csv', index=False)