#43k

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import pulp
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BATTERY OPTIMIZER - INTERACTION FEATURES")
print("="*60)

# ==========================================
# CONFIGURATION
# ==========================================
BATTERY_CAPACITY = 10.0
INITIAL_SOC = 5.0
MIN_TRANSACTION = 0.1

LAMBDA_GRID  = [50.0, 100.0, 200.0]
EPS_GRID     = [0.0, 1.0]
ALPHA_GRID   = [0.8, 0.9, 1.0]  

# ==========================================
# LOAD DATA
# ==========================================
print("\n[1/6] Loading data...")
df = pd.read_csv('/kaggle/input/dataset/Dataset.csv')
df['start_time_str'] = df['Time interval (CET/CEST)'].str.split(' - ').str[0]
df['datetime'] = pd.to_datetime(df['start_time_str'], dayfirst=True)
df = df.sort_values('datetime').reset_index(drop=True)

test_start = pd.Timestamp('2025-10-18 00:00:00')
val_start = test_start - pd.Timedelta(days=7)

train_df = df[df['datetime'] < val_start].copy()
val_df = df[(df['datetime'] >= val_start) & (df['datetime'] < test_start)].copy()
test_df = df[df['datetime'] >= test_start].copy()

print(f"   Train: {len(train_df)} intervals")
print(f"   Val:   {len(val_df)} intervals")
print(f"   Test:  {len(test_df)} intervals")

# ==========================================
# FEATURES
# ==========================================
print("\n[2/6] Engineering features...")

def add_features(df_subset):
    df_subset['hour'] = df_subset['datetime'].dt.hour
    df_subset['day_of_week'] = df_subset['datetime'].dt.dayofweek
    df_subset['month'] = df_subset['datetime'].dt.month
    df_subset['is_weekend'] = (df_subset['day_of_week'] >= 5).astype(int)
    df_subset['interval_of_day'] = df_subset['hour'] * 4 + df_subset['datetime'].dt.minute // 15
    
    # Basic cyclical
    df_subset['hour_sin'] = np.sin(2 * np.pi * df_subset['hour'] / 24)
    df_subset['hour_cos'] = np.cos(2 * np.pi * df_subset['hour'] / 24)
    df_subset['day_sin'] = np.sin(2 * np.pi * df_subset['day_of_week'] / 7)
    df_subset['day_cos'] = np.cos(2 * np.pi * df_subset['day_of_week'] / 7)
    df_subset['month_sin'] = np.sin(2 * np.pi * df_subset['month'] / 12)
    df_subset['month_cos'] = np.cos(2 * np.pi * df_subset['month'] / 12)
    
    # Peak indicators
    df_subset['is_evening_peak'] = ((df_subset['hour'] >= 17) & (df_subset['hour'] <= 20)).astype(int)
    df_subset['is_morning_low'] = ((df_subset['hour'] >= 3) & (df_subset['hour'] <= 6)).astype(int)
    
    # NEW: Polynomial features on hour (capture non-linear patterns)
    hour_normalized = df_subset['hour'] / 24.0
    df_subset['hour_squared'] = hour_normalized ** 2
    df_subset['hour_cubed'] = hour_normalized ** 3
    
    #  Day-Hour interactions
    df_subset['weekend_evening'] = df_subset['is_weekend'] * df_subset['is_evening_peak']
    df_subset['weekday_morning'] = (1 - df_subset['is_weekend']) * df_subset['is_morning_low']
    
    # Interaction between cyclical features
    df_subset['hour_day_interact'] = df_subset['hour_sin'] * df_subset['day_sin']
    df_subset['hour_day_interact2'] = df_subset['hour_cos'] * df_subset['day_cos']
    
    # Time of day buckets (more granular than just evening/morning)
    df_subset['is_night'] = ((df_subset['hour'] >= 0) & (df_subset['hour'] <= 5)).astype(int)
    df_subset['is_morning'] = ((df_subset['hour'] >= 6) & (df_subset['hour'] <= 11)).astype(int)
    df_subset['is_afternoon'] = ((df_subset['hour'] >= 12) & (df_subset['hour'] <= 17)).astype(int)
    df_subset['is_evening'] = ((df_subset['hour'] >= 18) & (df_subset['hour'] <= 23)).astype(int)
    
    return df_subset

train_df = add_features(train_df)
val_df = add_features(val_df)
test_df = add_features(test_df)

# Historical patterns (same as before)
hist_stats = (
    train_df.groupby(['day_of_week', 'interval_of_day'])['Price']
    .mean()
    .reset_index()
    .rename(columns={'Price': 'hist_avg_same_time'})
)

# More granular slot statistics
slot_stats = (
    train_df.groupby('interval_of_day')['Price']
    .agg(['mean', 'std', 'min', 'max'])
    .rename(columns={
        'mean': 'slot_mean_all',
        'std': 'slot_std_all',
        'min': 'slot_min_all',
        'max': 'slot_max_all'
    })
    .reset_index()
)

# Add slot range
slot_stats['slot_range_all'] = slot_stats['slot_max_all'] - slot_stats['slot_min_all']

sorted_means = slot_stats.set_index('interval_of_day')['slot_mean_all'].sort_values()
rank_map = {slot: i / (len(sorted_means) - 1) for i, slot in enumerate(sorted_means.index)}

def add_hist_features(df_split):
    df_split = df_split.merge(hist_stats, on=['day_of_week', 'interval_of_day'], how='left')
    df_split = df_split.merge(slot_stats, on='interval_of_day', how='left')
    df_split['slot_rank_all'] = df_split['interval_of_day'].map(rank_map)
    
    # Fill NaN
    df_split.fillna({
        'hist_avg_same_time': train_df['Price'].median(),
        'slot_mean_all': train_df['Price'].median(),
        'slot_std_all': train_df['Price'].std(),
        'slot_min_all': train_df['Price'].min(),
        'slot_max_all': train_df['Price'].max(),
        'slot_range_all': train_df['Price'].max() - train_df['Price'].min(),
        'slot_rank_all': 0.5
    }, inplace=True)
    
    return df_split

train_df = add_hist_features(train_df)
val_df = add_hist_features(val_df)
test_df = add_hist_features(test_df)

# ==========================================
# TRAIN MODELS
# ==========================================
print("\n[3/6] Training ensemble on enhanced features...")

feature_cols = [
    # Cyclical time
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'month_sin', 'month_cos',
    
    # Basic indicators
    'is_weekend', 'is_evening_peak', 'is_morning_low',
    
    # Polynomial
    'hour_squared', 'hour_cubed',
    
    # NEW: Interactions
    'weekend_evening', 'weekday_morning',
    'hour_day_interact', 'hour_day_interact2',
    
    # Time buckets
    'is_night', 'is_morning', 'is_afternoon', 'is_evening',
    
    # Historical
    'hist_avg_same_time',
    'slot_mean_all', 'slot_std_all', 'slot_min_all', 'slot_max_all',
    'slot_range_all', 'slot_rank_all'
]

print(f"   Total features: {len(feature_cols)}")

X_train = train_df[feature_cols]
y_train = train_df['Price']

xgb_model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, verbose=False)

lgb_model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=80,
    max_depth=7,
    subsample=0.8,
    random_state=456,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

cat_model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.05,
    depth=7,
    random_state=789,
    verbose=0
)
cat_model.fit(X_train, y_train)

# Weighted ensemble
X_val = val_df[feature_cols]
y_val = val_df['Price']

val_pred_xgb = xgb_model.predict(X_val)
val_pred_lgb = lgb_model.predict(X_val)
val_pred_cat = cat_model.predict(X_val)

mae_xgb = np.mean(np.abs(y_val - val_pred_xgb))
mae_lgb = np.mean(np.abs(y_val - val_pred_lgb))
mae_cat = np.mean(np.abs(y_val - val_pred_cat))

w_raw = np.array([1.0/max(mae_xgb, 1e-6), 1.0/max(mae_lgb, 1e-6), 1.0/max(mae_cat, 1e-6)])
w = w_raw / w_raw.sum()
w_xgb, w_lgb, w_cat = w.tolist()

val_pred = w_xgb * val_pred_xgb + w_lgb * val_pred_lgb + w_cat * val_pred_cat
val_mae = np.mean(np.abs(y_val - val_pred))

print(f"   Ensemble MAE: €{val_mae:.2f}")
print(f"   (Previous best: €673)")

if val_mae < 673:
    print(f"   ✓ MAE improved by €{673 - val_mae:.2f}!")
else:
    print(f"   ⚠ MAE slightly worse")

# ==========================================
# LP OPTIMIZER
# ==========================================
def optimize_day(prices, lambda_soc=200.0, eps=0.0):
    n = len(prices)
    if n != 96:
        return [0.0] * n
    
    prob = pulp.LpProblem("Battery", pulp.LpMaximize)
    c = pulp.LpVariable.dicts("C", range(n), lowBound=0)
    d = pulp.LpVariable.dicts("D", range(n), lowBound=0)
    s = pulp.LpVariable.dicts("S", range(n), lowBound=0, upBound=BATTERY_CAPACITY)
    
    prob += (
        pulp.lpSum([(d[t] - c[t]) * prices[t] for t in range(n)])
        - lambda_soc * s[n-1]
        - eps * pulp.lpSum([c[t] + d[t] for t in range(n)])
    )
    
    for t in range(n):
        if t == 0:
            prob += s[t] == INITIAL_SOC + c[t] - d[t]
        else:
            prob += s[t] == s[t-1] + c[t] - d[t]
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    return [
        (pulp.value(c[t]) - pulp.value(d[t])) if pulp.value(c[t]) is not None and abs(pulp.value(c[t]) - pulp.value(d[t])) >= MIN_TRANSACTION else 0.0
        for t in range(n)
    ]

def evaluate_config(df_eval, ml_pred, lam, eps, alpha):
    df_eval = df_eval.copy()
    df_eval['ml_pred'] = ml_pred
    df_eval['pred_blend'] = alpha * df_eval['ml_pred'] + (1 - alpha) * df_eval['hist_avg_same_time']
    df_eval['date'] = df_eval['datetime'].dt.date
    
    total = 0.0
    for date in sorted(df_eval['date'].unique()):
        day_df = df_eval[df_eval['date'] == date]
        if len(day_df) != 96:
            continue
        
        actions = optimize_day(day_df['pred_blend'].values, lam, eps)
        profit = sum(-a * p for a, p in zip(actions, day_df['Price']))
        soc_end = INITIAL_SOC + sum(actions)
        total += profit + soc_end * day_df['Price'].min()
    
    return total

# ==========================================
# TUNE
# ==========================================
print("\n[4/6] Tuning with enhanced features...")

val_df['ml_pred'] = val_pred
best_val_profit = -float('inf')
best_config = None

for lam in LAMBDA_GRID:
    for eps in EPS_GRID:
        for alpha in ALPHA_GRID:
            val_profit = evaluate_config(val_df, val_pred, lam, eps, alpha)
            if val_profit > best_val_profit:
                best_val_profit = val_profit
                best_config = (lam, eps, alpha)
                print(f"   NEW BEST: λ={lam:>5.0f} ε={eps:>3.0f} α={alpha:.1f} → €{val_profit:>10,.0f}")

BEST_LAMBDA, BEST_EPS, BEST_ALPHA = best_config

print(f"\n{'='*60}")
print(f"Best config: λ={BEST_LAMBDA}, ε={BEST_EPS}, α={BEST_ALPHA}")
print(f"Val profit: €{best_val_profit:,.0f}")
print(f"(Previous: €462,434 with α=0.8)")
print(f"{'='*60}")

# ==========================================
# TEST
# ==========================================
print("\n[5/6] Test evaluation...")

X_test = test_df[feature_cols]
test_pred = w_xgb * xgb_model.predict(X_test) + w_lgb * lgb_model.predict(X_test) + w_cat * cat_model.predict(X_test)
test_profit = evaluate_config(test_df, test_pred, BEST_LAMBDA, BEST_EPS, BEST_ALPHA)

print(f"   Test profit: €{test_profit:,.0f}")
print(f"   Val/Test ratio: {best_val_profit/test_profit:.2f}x")

# ==========================================
# FUTURE
# ==========================================
print("\n[6/6] Generating submission...")

future_dates = pd.date_range(start='2025-10-26 00:00:00', end='2025-11-02 23:45:00', freq='15min')
future_df = pd.DataFrame({'datetime': future_dates})
future_df = add_features(future_df)
future_df = add_hist_features(future_df)

X_future = future_df[feature_cols]
future_ml_pred = w_xgb * xgb_model.predict(X_future) + w_lgb * lgb_model.predict(X_future) + w_cat * cat_model.predict(X_future)
future_df['pred_blend'] = BEST_ALPHA * future_ml_pred + (1 - BEST_ALPHA) * future_df['hist_avg_same_time']
future_df['date'] = future_df['datetime'].dt.date

future_actions = []
for date in sorted(future_df['date'].unique()):
    day_df = future_df[future_df['date'] == date]
    actions = optimize_day(day_df['pred_blend'].values, BEST_LAMBDA, BEST_EPS)
    future_actions.extend(actions)
    
    n_trades = sum(1 for a in actions if abs(a) >= MIN_TRANSACTION)
    print(f"   {pd.Timestamp(date).strftime('%A, %b %d')}: {n_trades} trades")

future_df['time_str'] = future_df['datetime'].dt.strftime('%d.%m.%Y %H:%M')
future_df['end_time'] = (future_df['datetime'] + pd.Timedelta(minutes=15)).dt.strftime('%d.%m.%Y %H:%M')
future_df['time_interval'] = future_df['time_str'] + ' - ' + future_df['end_time']

submission = pd.DataFrame({
    "Time interval (CET/CEST)": future_df['time_interval'].values,
    "Position": future_actions
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
