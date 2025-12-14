'''
High-Performance Churn Pipeline (Ensemble + 35% Balance + Direct Optuna)
=========================================================================
Strategy:
- Train: Sliding Window Augmentation (Obs=20d, Stride=7d)
- Balancing: Undersample non-churners to 35% churn rate
- Model: Seed Averaging Ensemble (5 seeds)
- Features: Rolling Windows, Time-of-Day, Engagement Trends
- Optimization: Vectorized Time-Slicing
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import re
import sys
import optuna
from optuna.samplers import TPESampler

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CONF = {
    'OBS_DAYS': 20,
    'PRED_DAYS': 10,
    'STRIDE': 7,
    'PRED_START_DATE': datetime(2018, 11, 20),
    'TOP_K': 20,
    'N_TRIALS': 150,
    'TIMEOUT': 2400,
    'TARGET_CHURN_RATIO': 0.35,     # 35% Churners
    'SEEDS': [42, 123, 456, 789, 1010]  # Ensemble Seeds
}

class ChurnPipeline:
    def __init__(self):
        self.models = [] 
        self.selected_features = None
        self.best_threshold = 0.5
        self.inactive_churn_rate = 0.3

    def _get_slice(self, df, start_dt, end_dt, sorted_times):
        '''Fastest possible dataframe slicing using binary search.'''
        start_idx = np.searchsorted(sorted_times, np.datetime64(start_dt))
        end_idx = np.searchsorted(sorted_times, np.datetime64(end_dt))
        return df.iloc[start_idx:end_idx].copy()

    def extract_features(self, df_slice, window_start, window_end):
        '''Extracts rich feature set efficiently.'''
        if len(df_slice) == 0: 
            return pd.DataFrame()
        
        # --- PRE-COMPUTATION ---
        df_slice['hour'] = df_slice['time'].dt.hour
        df_slice['dow'] = df_slice['time'].dt.dayofweek
        
        # Base Aggregates
        features = df_slice.groupby('userId').agg({
            'sessionId': 'nunique',
            'length': 'sum',
            'level': 'last', 
            'registration': 'first'
        }).reset_index()
        
        features.rename(
            columns={
                'sessionId': 'total_sessions', 
                'length': 'total_length'}, 
                inplace=True
        )
        features['level'] = features['level'].map(
            {'free': 0, 'paid': 1}
        ).fillna(0)
        
        # Tenure
        features['days_since_reg'] = (
            window_end - features['registration']
        ).dt.total_seconds() / 86400
        features.drop('registration', axis=1, inplace=True)

        # --- TIME OF DAY FEATURES ---
        hours = df_slice['hour'].values
        dows = df_slice['dow'].values
        
        df_slice['is_morning'] = ((hours >= 6) & (hours < 12)).astype(int)
        df_slice['is_afternoon'] = ((hours >= 12) & (hours < 18)).astype(int)
        df_slice['is_evening'] = ((hours >= 18) & (hours < 24)).astype(int)
        df_slice['is_night'] = ((hours >= 0) & (hours < 6)).astype(int)
        df_slice['is_weekend'] = (dows >= 5).astype(int)
        
        time_stats = df_slice.groupby('userId')[
            ['is_morning', 'is_afternoon', 'is_evening', 'is_night', 'is_weekend']
        ].sum().reset_index()
        features = features.merge(time_stats, on='userId', how='left')

        # --- SPECIFIC PAGE COUNTS ---
        target_pages = [
            'Thumbs Up', 'Thumbs Down', 'Roll Advert', 'Add to Playlist', 
            'Upgrade', 'Downgrade', 'Submit Downgrade', 'Error'
        ]
        page_subset = df_slice[df_slice['page'].isin(target_pages)]
        if not page_subset.empty:
            page_counts = page_subset.pivot_table(
                index='userId', 
                columns='page', 
                values='time', 
                aggfunc='count', 
                fill_value=0
            )
            page_counts.columns = [
                f"count_{c.replace(' ', '_')}" for c in page_counts.columns
            ]
            features = features.merge(page_counts, on='userId', how='left')

        # --- ROLLING WINDOWS LOOP ---
        windows = [1, 3, 7, 14]
        for w in windows:
            w_start = window_end - timedelta(days=w)
            w_df = df_slice[df_slice['time'] >= w_start]
            
            if w_df.empty:
                features[f'sessions_{w}d'] = 0
                features[f'songs_{w}d'] = 0
                continue
                
            w_stats = w_df.groupby('userId').agg({
                'sessionId': 'nunique',
                'song': 'nunique',
                'length': 'sum',
                'page': 'count'
            }).reset_index()
            
            w_stats.columns = [
                'userId', 
                f'sessions_{w}d', 
                f'unique_songs_{w}d', 
                f'duration_{w}d', 
                f'actions_{w}d'
            ]
            features = features.merge(w_stats, on='userId', how='left')
            
        features = features.fillna(0)
        
        # --- RATIOS ---
        features['songs_per_session'] = features['unique_songs_7d'] / (
            features['sessions_7d'] + 1
        )
        features['avg_session_duration'] = features['total_length'] / (
            features['total_sessions'] + 1
        )
        features['weekend_ratio'] = features['is_weekend'] / (
            features['total_sessions'] + 1
        )
        features['short_term_drop'] = features['sessions_3d'] - (
            features['sessions_7d'] / 2.3
        )
        features['has_activity'] = 1
            
        return features

    def extract_labels(self, df_slice):
        '''Extracts churn labels.'''
        if len(df_slice) == 0: 
            return pd.DataFrame()
        
        churners = df_slice[
            df_slice['page'] == 'Cancellation Confirmation'
        ]['userId'].unique()
        all_users = df_slice['userId'].unique()
        return pd.DataFrame({
            'userId': all_users, 
            'churned': np.isin(all_users, churners).astype(int)
        })

    def generate_training_data(self, df):
        print('\n Pipeline: Generating Sliding Window Features')
        
        df = df.sort_values('time').reset_index(drop=True)
        sorted_times = df['time'].values
        
        min_date, max_date = df['time'].min(), df['time'].max()
        current = min_date
        X_list, y_list = [], []
        window_count = 0
        
        while True:
            obs_start = current
            obs_end = obs_start + timedelta(days=CONF['OBS_DAYS'])
            pred_start = obs_end
            pred_end = pred_start + timedelta(days=CONF['PRED_DAYS'])
            
            if pred_end > max_date: 
                break
            
            pred_slice = self._get_slice(
                df, pred_start, pred_end, sorted_times
            )
            if len(pred_slice) > 0:
                obs_slice = self._get_slice(
                    df, obs_start, obs_end, sorted_times
                )
                if len(obs_slice) > 0:
                    features = self.extract_features(
                        obs_slice, obs_start, obs_end
                    )
                    labels = self.extract_labels(pred_slice)
                    if not features.empty and not labels.empty:
                        data = features.merge(
                            labels, on='userId', how='inner'
                        )
                        if len(data) > 0:
                            data['has_activity'] = data[
                                'has_activity'
                            ].fillna(0)
                            data = data.fillna(0)
                            X_list.append(
                                data.drop(['userId', 'churned'], axis=1)
                            )
                            y_list.append(data['churned'])
            
            current += timedelta(days=CONF['STRIDE'])
            window_count += 1
            sys.stdout.write(f'\r  Processed {window_count} windows')
            sys.stdout.flush()

        print('\n  Stacking data')
        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True)
        
        inactive_mask = X['has_activity'] == 0
        if inactive_mask.sum() > 0:
            self.inactive_churn_rate = y[inactive_mask].mean()
            
        # --- BALANCING LOGIC (35%) ---
        print(
            f'\n  Balancing Classes (Target: ',
            f"{CONF['TARGET_CHURN_RATIO']:.0%} churners)")
        print(f'  Original count: {len(X)} (Churn rate: {y.mean():.2%})')
        
        X_churn = X[y == 1]
        y_churn = y[y == 1]
        X_non = X[y == 0]
        y_non = y[y == 0]
        
        n_churn = len(X_churn)
        # Calculate N to keep for non-churners
        n_keep = int(n_churn * (1 - CONF['TARGET_CHURN_RATIO']) / 
                     CONF['TARGET_CHURN_RATIO'])
        
        if n_keep < len(X_non):
            print(f'  Downsampling non-churners from {len(X_non)} to {n_keep}')
            indices = np.random.choice(len(X_non), n_keep, replace=False)
            X_non = X_non.iloc[indices]
            y_non = y_non.iloc[indices]
        
        X = pd.concat([X_churn, X_non])
        y = pd.concat([y_churn, y_non])
        idx = np.random.permutation(len(X))
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        
        print(f'  Balanced count: {len(X)} (Churn rate: {y.mean():.2%})')
        return X, y

    def select_features(self, X, y):
        print(f"\n Pipeline: Selecting Top {CONF['TOP_K']} Features")
        special = ['has_activity']
        X_sel = X.drop([c for c in special if c in X.columns], axis=1)
        selector = SelectKBest(mutual_info_classif, k=CONF['TOP_K'])
        selector.fit(X_sel, y)
        cols = X_sel.columns[selector.get_support()].tolist() + special
        self.selected_features = [c for c in cols if c in X.columns]
        print(f'  Selected {len(self.selected_features)} features.')
        return X[self.selected_features]

    def optimize_and_train(self, X, y):
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators', 150, 400, step=50
                ),
                'max_depth': trial.suggest_int('max_depth', 5, 9),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.02, 0.15, log=True
                ),
                'min_child_weight': trial.suggest_int(
                    'min_child_weight', 1, 6
                ),
                'gamma': trial.suggest_float('gamma', 0.0, 0.3),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree', 0.7, 0.95
                ),
                'scale_pos_weight': trial.suggest_float(
                    'scale_pos_weight', 1, 10
                ),
                'objective': 'binary:logistic', 'eval_metric': 'auc', 
                'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42
            }
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            for tr, val in skf.split(X, y):
                m = XGBClassifier(**params)
                m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
                cv_scores.append(
                    roc_auc_score(
                        y.iloc[val], 
                        m.predict_proba(X.iloc[val])[:, 1]
                    )
                )
            return np.mean(cv_scores)

        print('\n Pipeline: Optimizing Hyperparameters')
        study = optuna.create_study(
            direction='maximize', 
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            objective, 
            n_trials=CONF['N_TRIALS'], 
            timeout=CONF['TIMEOUT']
        )
        best_params = study.best_params

        best_params.update({
            'objective': 'binary:logistic', 
            'eval_metric': 
            'auc', 'tree_method': 
            'hist', 'n_jobs': -1
        })
        
        # --- THRESHOLD OPTIMIZATION (Using Single Seed) ---
        print('\n  Optimizing Threshold (Single Seed)')
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        model = XGBClassifier(**best_params, random_state=42)
        model.fit(X_tr, y_tr)
        val_proba = model.predict_proba(X_val)[:, 1]
        p, r, t = precision_recall_curve(y_val, val_proba)
        
        valid_idx = np.where(r >= 0.55)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(p[valid_idx])]
            self.best_threshold = t[best_idx] if best_idx < len(t) else 0.5
        else:
            self.best_threshold = t[np.argmax(2*(p*r)/(p+r+1e-10))]
        print(f'  Optimal Threshold: {self.best_threshold:.3f}')

        # --- SEED AVERAGING TRAINING ---
        print(
            f"\n Pipeline: Training {len(CONF['SEEDS'])} Seeds on Full Data"
        )
        self.models = []
        for seed in CONF['SEEDS']:
            print(f'  Training seed {seed}')
            params = best_params.copy()
            params['random_state'] = seed
            m = XGBClassifier(**params)
            m.fit(X, y)
            self.models.append(m)

    def predict(self, test_df):
        print('\n Pipeline: Predicting Test Data')
        obs_end = CONF['PRED_START_DATE']
        obs_start = obs_end - timedelta(days=CONF['OBS_DAYS'])
        
        test_df = test_df.sort_values('time').reset_index(drop=True)
        sorted_times = test_df['time'].values
        obs_slice = self._get_slice(test_df, obs_start, obs_end, sorted_times)
        feats = self.extract_features(obs_slice, obs_start, obs_end)
        
        all_users = pd.DataFrame({'userId': test_df['userId'].unique()})
        data = all_users.merge(feats, on='userId', how='left')
        data['has_activity'] = data['has_activity'].fillna(0)
        
        ids = data['userId']
        is_inactive = data['has_activity'] == 0
        X_test = data.drop(['userId'], axis=1)
        
        X_test = pd.get_dummies(X_test)
        X_test = X_test.reindex(
            columns=self.selected_features, fill_value=0
        ).fillna(0)
        
        # --- ENSEMBLE PREDICTION ---
        test_proba_list = []
        for m in self.models:
            test_proba_list.append(m.predict_proba(X_test)[:, 1])
            
        avg_probs = np.mean(test_proba_list, axis=0)
        preds = (avg_probs >= self.best_threshold).astype(int)
        
        inactive_val = int(self.inactive_churn_rate >= 0.5)
        preds[is_inactive] = inactive_val
        
        # Average Feature Importance
        avg_imp = np.mean(
            [m.feature_importances_ for m in self.models], axis=0
        )
        
        return pd.DataFrame({'id': ids, 'target': preds}), X_test.columns, avg_imp

def main():
    print('='*100 + '\nCHURN PIPELINE (ENSEMBLE + 35% BALANCE)\n' + '='*100)
    pipeline = ChurnPipeline()
    
    print(' Loading Data')
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    for df in [train_df, test_df]:
        df['time'] = pd.to_datetime(df['time'])
        df['registration'] = pd.to_datetime(df['registration'])
    
    X_train, y_train = pipeline.generate_training_data(train_df)
    
    cat_cols = X_train.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X_train_sel = pipeline.select_features(X_train, y_train)
    pipeline.optimize_and_train(X_train_sel, y_train)
    
    submission, cols, imps = pipeline.predict(test_df)
    submission.to_csv('churn_predictions_pipeline.csv', index=False)
    print("\n Saved to 'churn_predictions_pipeline.csv'")
    
    imp_df = pd.DataFrame(
        {'feature': cols, 'importance': imps}
    ).sort_values('importance', ascending=False)
    print('\n Top Features (Averaged):')
    print(imp_df.head(15))

if __name__ == '__main__':
    main()
