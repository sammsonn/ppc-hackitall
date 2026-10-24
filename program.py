import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import pulp
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BATTERY OPTIMIZER V5 - ROBUST (NO OVERFITTING)")
print("="*60)

# ==========================================
# CONFIGURATION
# ==========================================
BATTERY_CAPACITY = 10.0
INITIAL_SOC = 5.0
MIN_TRANSACTION = 0.1
INTERVALS_PER_DAY = 96

# CONSERVATIVE GRID - don't overfit!
LAMBDA_GRID = [100.0, 200.0, 400.0]
EPS_GRID = [0.0, 1.0]
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8]  

# ==========================================
# LOAD DATA
# ==========================================
print("\n[1/7] Loading data...")
df = pd.read_csv('/kaggle/input/dataset/Dataset.csv')
df['start_time_str'] = df['Time interval (CET/CEST)'].str.split(' - ').str[0]
df['datetime'] = pd.to_datetime(df['start_time_str'], dayfirst=True)
df = df.sort_values('datetime').reset_index(drop=True)

print(f"   Total intervals: {len(df)}")

# Test period 
test_start = pd.Timestamp('2025-10-18 00:00:00')
test_end = pd.Timestamp('2025-10-25 23:45:00')

# Validation period (1 week before test)
val_start = test_start - pd.Timedelta(days=7)
val_end = test_start - pd.Timedelta(minutes=15)

# Training (everything before validation)
train_mask = df['datetime'] < val_start
val_mask = (df['datetime'] >= val_start) & (df['datetime'] <= val_end)
test_mask = (df['datetime'] >= test_start) & (df['datetime'] <= test_end)

train_df = df[train_mask].copy()
val_df = df[val_mask].copy()
test_df = df[test_mask].copy()

print(f"   Train: {len(train_df)} intervals ({train_df['datetime'].min().date()} to {train_df['datetime'].max().date()})")
print(f"   Val:   {len(val_df)} intervals ({val_df['datetime'].min().date()} to {val_df['datetime'].max().date()})")
print(f"   Test:  {len(test_df)} intervals ({test_df['datetime'].min().date()} to {test_df['datetime'].max().date()})")

# ==========================================
# FEATURES - CALCULATED ON TRAIN ONLY
# ==========================================
print("\n[3/7] Engineering features (TRAIN ONLY)...")

def add_time_features(df_subset):
    df_subset['hour'] = df_subset['datetime'].dt.hour
    df_subset['day_of_week'] = df_subset['datetime'].dt.dayofweek
    df_subset['month'] = df_subset['datetime'].dt.month
    df_subset['is_weekend'] = (df_subset['day_of_week'] >= 5).astype(int)
    df_subset['interval_of_day'] = df_subset['hour'] * 4 + df_subset['datetime'].dt.minute // 15
    
    df_subset['hour_sin'] = np.sin(2 * np.pi * df_subset['hour'] / 24)
    df_subset['hour_cos'] = np.cos(2 * np.pi * df_subset['hour'] / 24)
    df_subset['day_sin'] = np.sin(2 * np.pi * df_subset['day_of_week'] / 7)
    df_subset['day_cos'] = np.cos(2 * np.pi * df_subset['day_of_week'] / 7)
    df_subset['month_sin'] = np.sin(2 * np.pi * df_subset['month'] / 12)
    df_subset['month_cos'] = np.cos(2 * np.pi * df_subset['month'] / 12)
    
    df_subset['is_evening_peak'] = ((df_subset['hour'] >= 17) & (df_subset['hour'] <= 20)).astype(int)
    df_subset['is_morning_low'] = ((df_subset['hour'] >= 3) & (df_subset['hour'] <= 6)).astype(int)
    return df_subset

# Add time features to all splits
for df_split in [train_df, val_df, test_df]:
    add_time_features(df_split)

# Calculate historical stats ONLY from training data
print("   Calculating historical patterns from TRAIN only...")
hist_stats = (
    train_df.groupby(['day_of_week', 'interval_of_day'])['Price']
    .mean()
    .reset_index()
    .rename(columns={'Price': 'hist_avg_same_time'})
)

slot_stats = (
    train_df.groupby('interval_of_day')['Price']
    .agg(['mean', 'std'])
    .rename(columns={'mean': 'slot_mean_all', 'std': 'slot_std_all'})
    .reset_index()
)

sorted_means = slot_stats.set_index('interval_of_day')['slot_mean_all'].sort_values()
rank_map = {slot: i / (len(sorted_means) - 1) for i, slot in enumerate(sorted_means.index)}

# Apply to all splits
for df_split in [train_df, val_df, test_df]:
    df_split = df_split.merge(hist_stats, on=['day_of_week', 'interval_of_day'], how='left')
    df_split = df_split.merge(slot_stats, on='interval_of_day', how='left')
    df_split['slot_rank_all'] = df_split['interval_of_day'].map(rank_map)
    df_split.fillna(method='ffill', inplace=True)
    df_split.fillna(0, inplace=True)

# Update the original dataframes
train_df = train_df.merge(hist_stats, on=['day_of_week', 'interval_of_day'], how='left')
train_df = train_df.merge(slot_stats, on='interval_of_day', how='left')
train_df['slot_rank_all'] = train_df['interval_of_day'].map(rank_map)

val_df = val_df.merge(hist_stats, on=['day_of_week', 'interval_of_day'], how='left')
val_df = val_df.merge(slot_stats, on='interval_of_day', how='left')
val_df['slot_rank_all'] = val_df['interval_of_day'].map(rank_map)

test_df = test_df.merge(hist_stats, on=['day_of_week', 'interval_of_day'], how='left')
test_df = test_df.merge(slot_stats, on='interval_of_day', how='left')
test_df['slot_rank_all'] = test_df['interval_of_day'].map(rank_map)

# Fill any remaining NaNs
for df_split in [train_df, val_df, test_df]:
    df_split.fillna(method='ffill', inplace=True)
    df_split.fillna(0, inplace=True)

# ==========================================
# TRAIN MODELS
# ==========================================
print("\n[4/7] Training ensemble...")

feature_cols = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'is_evening_peak', 'is_morning_low',
    'hist_avg_same_time',
    'slot_mean_all', 'slot_std_all', 'slot_rank_all'
]

X_train = train_df[feature_cols]
y_train = train_df['Price']
X_val = val_df[feature_cols]
y_val = val_df['Price']

print("   XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,  
    learning_rate=0.1,
    max_depth=6,  
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, verbose=False)

print("   LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    random_state=456,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

print("   CatBoost...")
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_state=789,
    verbose=0
)
cat_model.fit(X_train, y_train)

# Validate predictions
val_pred = (
    0.4 * xgb_model.predict(X_val) +
    0.35 * lgb_model.predict(X_val) +
    0.25 * cat_model.predict(X_val)
)

val_mae = np.mean(np.abs(y_val - val_pred))
val_mape = np.mean(np.abs(y_val - val_pred) / np.abs(y_val)) * 100

print(f"\n   Validation metrics:")
print(f"   MAE:  €{val_mae:.2f}")
print(f"   MAPE: {val_mape:.1f}%")

if val_mape > 50:
    print(f"   ⚠ WARNING: High MAPE - predictions may be unreliable")

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
        (pulp.value(c[t]) - pulp.value(d[t])) if abs(pulp.value(c[t]) - pulp.value(d[t])) >= MIN_TRANSACTION else 0.0
        for t in range(n)
    ]

def evaluate_config(df_eval, pred_col, lambda_soc, eps, alpha):
    df_eval = df_eval.copy()
    df_eval['pred_blend'] = alpha * df_eval[pred_col] + (1 - alpha) * df_eval['hist_avg_same_time']
    df_eval['date'] = df_eval['datetime'].dt.date
    
    total_profit = 0
    for date in sorted(df_eval['date'].unique()):
        day_df = df_eval[df_eval['date'] == date]
        if len(day_df) != 96:
            continue
        
        actions = optimize_day(day_df['pred_blend'].values, lambda_soc, eps)
        profit = sum(-a * p for a, p in zip(actions, day_df['Price']))
        
        # Add EOD settlement
        soc_end = INITIAL_SOC + sum(actions)
        profit += soc_end * day_df['Price'].min()
        
        total_profit += profit
    
    return total_profit

# ==========================================
# TUNE ON VALIDATION 
# ==========================================
print("\n[5/7] Tuning on VALIDATION set...")

# Get validation predictions
val_df['ml_pred'] = val_pred

best_val_profit = -float('inf')
best_config = None

print(f"   Testing {len(LAMBDA_GRID) * len(EPS_GRID) * len(ALPHA_GRID)} configs...")

for lam in LAMBDA_GRID:
    for eps in EPS_GRID:
        for alpha in ALPHA_GRID:
            val_profit = evaluate_config(val_df, 'ml_pred', lam, eps, alpha)
            
            if val_profit > best_val_profit:
                best_val_profit = val_profit
                best_config = (lam, eps, alpha)
                print(f"   NEW BEST: λ={lam:>5.0f} ε={eps:>3.0f} α={alpha:.2f} → Val profit: €{val_profit:>10,.2f}")

BEST_LAMBDA, BEST_EPS, BEST_ALPHA = best_config

print(f"\n{'='*60}")
print("BEST CONFIG (from validation):")
print(f"{'='*60}")
print(f"  Lambda:     {BEST_LAMBDA}")
print(f"  Epsilon:    {BEST_EPS}")
print(f"  Alpha:      {BEST_ALPHA}")
print(f"  Val profit: €{best_val_profit:,.2f}")
print(f"{'='*60}")

# ==========================================
# EVALUATE ON TEST 
# ==========================================
print("\n[6/7] Evaluating on test set (for info only)...")

X_test = test_df[feature_cols]
test_pred = (
    0.4 * xgb_model.predict(X_test) +
    0.35 * lgb_model.predict(X_test) +
    0.25 * cat_model.predict(X_test)
)
test_df['ml_pred'] = test_pred

test_profit = evaluate_config(test_df, 'ml_pred', BEST_LAMBDA, BEST_EPS, BEST_ALPHA)

print(f"   Test profit (with best config): €{test_profit:,.2f}")
print(f"   Val/Test ratio: {best_val_profit/test_profit:.2f}x")

if abs(best_val_profit/test_profit - 1.0) > 0.5:
    print(f"   ⚠ WARNING: Large val/test mismatch - may indicate instability")

# ==========================================
# FUTURE PREDICTIONS
# ==========================================
print("\n[7/7] Generating submission...")

future_dates = pd.date_range(start='2025-10-26 00:00:00', end='2025-11-02 23:45:00', freq='15min')
future_df = pd.DataFrame({'datetime': future_dates})

add_time_features(future_df)

future_df = future_df.merge(hist_stats, on=['day_of_week', 'interval_of_day'], how='left')
future_df = future_df.merge(slot_stats, on='interval_of_day', how='left')
future_df['slot_rank_all'] = future_df['interval_of_day'].map(rank_map)
future_df.fillna(method='ffill', inplace=True)
future_df.fillna(0, inplace=True)

X_future = future_df[feature_cols]
future_pred = (
    0.4 * xgb_model.predict(X_future) +
    0.35 * lgb_model.predict(X_future) +
    0.25 * cat_model.predict(X_future)
)

future_df['pred_blend'] = BEST_ALPHA * future_pred + (1 - BEST_ALPHA) * future_df['hist_avg_same_time']
future_df['date'] = future_df['datetime'].dt.date

future_actions = []
for date in sorted(future_df['date'].unique()):
    day_df = future_df[future_df['date'] == date]
    actions = optimize_day(day_df['pred_blend'].values, BEST_LAMBDA, BEST_EPS)
    future_actions.extend(actions)
    
    n_trades = sum(1 for a in actions if abs(a) >= MIN_TRANSACTION)
    print(f"   {pd.Timestamp(date).strftime('%A, %b %d')}: {n_trades} trades")

# Create submission
future_df['time_str'] = future_df['datetime'].dt.strftime('%d.%m.%Y %H:%M')
future_df['end_time'] = (future_df['datetime'] + pd.Timedelta(minutes=15)).dt.strftime('%d.%m.%Y %H:%M')
future_df['time_interval'] = future_df['time_str'] + ' - ' + future_df['end_time']

submission = pd.DataFrame({
    "Time interval (CET/CEST)": future_df['time_interval'].values,
    "Position": future_actions
})

print(f"\n{'='*60}")
print(f"Total trades: {(submission['Position'] != 0).sum()}")
print(f"{'='*60}")

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\n✓ submission.csv saved (validated on proper holdout set)")
print("="*60)
