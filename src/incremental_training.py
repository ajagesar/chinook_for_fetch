"""
incremental_training.py

Incrementally grows the training set one db_origin at a time and fully
retrains every model at each step.  The test set is fixed (all origins)
so every step is evaluated on the same held-out data — the only thing
that changes is how much training data the model has seen.

Pipeline per step
-----------------
1. Slice X_train / y_train to the cumulative origins so far.
2. Fit imputer on that slice; transform slice + full X_test.
3. (Optional) SMOTE on the imputed training slice.
4. Train every model that exists in config['model_algorithms'].
5. Predict on the (identically-imputed) X_test.
6. Record AUC, F1, Brier score.

Results are written to
    data/final/incremental_results.yaml
which plotting.py reads to produce the performance-vs-data-source chart.
"""

import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from pathlib import Path

import torch
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer          # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, f1_score, brier_score_loss
from imblearn.over_sampling import SMOTE
from skorch import NeuralNetBinaryClassifier

from utils.helpers import ROOT, config
from utils.torchneuralnet import PyTorchNN

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    logger.info("=" * 60)
    logger.info("INCREMENTAL TRAINING — starting")
    logger.info("=" * 60)

    # --- 1. Load everything we need upfront --- #
    (
        X_train_raw,          # unimputed, contains db_origin
        y_train,
        X_test_raw,           # unimputed, no db_origin needed (fixed)
        y_test,
        db_origin_train,      # Series aligned with X_train_raw
    ) = load_all_data()

    # --- 2. Decide the order in which origins are added --- #
    origin_order = get_origin_order(db_origin_train)
    logger.info(f"Origin order (largest first): {origin_order}")

    # --- 3. Drop db_origin from the feature matrix before any modelling --- #
    # db_origin is metadata, not a feature.  It may or may not be present
    # depending on whether feature selection already removed it.
    X_train_raw = _drop_if_present(X_train_raw, 'db_origin')
    X_train_raw = _drop_if_present(X_train_raw, 'db origin')
    X_test_raw  = _drop_if_present(X_test_raw,  'db_origin')
    X_test_raw  = _drop_if_present(X_test_raw,  'db origin')

    # --- 4. Load hyper-parameter files once (shared across steps) --- #
    hyperparams = load_all_hyperparams()

    # --- 5. Cumulative loop --- #
    all_results = {}                          # { model_label: [ {step_info} ] }
    cumulative_mask = pd.Series(False, index=X_train_raw.index)

    for step_idx, origin in enumerate(origin_order):
        cumulative_mask = cumulative_mask | (db_origin_train == origin)
        origins_so_far  = origin_order[: step_idx + 1]
        step_label      = f"step_{step_idx}"

        logger.info("")
        logger.info(f"--- Step {step_idx}: adding '{origin}' "
                    f"(origins so far: {origins_so_far}) ---")

        # slice training data for this step
        X_train_step = X_train_raw.loc[cumulative_mask].copy()
        y_train_step = y_train.loc[cumulative_mask].copy()

        n_train = len(X_train_step)
        n_pos  = int(y_train_step.sum())
        logger.info(f"    Training samples: {n_train}  (positive: {n_pos})")

        # need at least two classes to train
        if y_train_step.nunique() < 2:
            logger.warning("    Only one class in training slice — skipping step.")
            continue

        # --- 5a. Imputation (fit on THIS step's train only) --- #
        X_train_imp, X_test_imp = impute_step(X_train_step, X_test_raw)

        # --- 5b. Optional SMOTE --- #
        if config.get('oversample', False):
            X_train_imp, y_train_step = apply_smote(X_train_imp, y_train_step)
            logger.info(f"    After SMOTE: {len(X_train_imp)} samples")

        # --- 5c. Train & evaluate every model --- #
        step_results = train_and_evaluate_all(
            X_train_imp=X_train_imp,
            X_train_raw_step=X_train_step,   # unimputed, for xgb_unimputed
            y_train=y_train_step,
            X_test_imp=X_test_imp,
            y_test=y_test,
            hyperparams=hyperparams,
        )

        # --- 5d. Store results --- #
        for model_label, metrics in step_results.items():
            metrics['step_idx']          = step_idx
            metrics['origins_included']  = origins_so_far
            metrics['n_train']           = n_train
            all_results.setdefault(model_label, []).append(metrics)

    # --- 6. Persist --- #
    out_path = ROOT + 'data/final/incremental_results.yaml'
    with open(out_path, 'w') as f:
        yaml.safe_dump(all_results, f, default_flow_style=False)

    logger.info("")
    logger.info(f"Results saved to {out_path}")
    logger.info("=" * 60)
    logger.info("INCREMENTAL TRAINING — done")
    logger.info("=" * 60)


# ===========================================================================
# Data loading
# ===========================================================================

def load_all_data():
    """Return raw (unimputed) train/test splits plus the db_origin column.

    We load from the *pre-imputation* CSVs so that we can re-fit the imputer
    on each incremental training slice independently.  db_origin comes from
    test_data_complete.csv (or an equivalent file that was saved during
    splitting with all metadata intact).
    """

    X_train = pd.read_csv(ROOT + 'data/intermediate/X_train.csv')
    y_train = pd.read_csv(ROOT + 'data/intermediate/y_train.csv')['in_icu_death']
    X_test  = pd.read_csv(ROOT + 'data/intermediate/X_test.csv')
    y_test  = pd.read_csv(ROOT + 'data/intermediate/y_test.csv')['in_icu_death']

    # db_origin for the training set.  This column must exist in one of:
    #   • X_train.csv itself (if splitting.py kept it)
    #   • a separate metadata file saved alongside the split
    # We try X_train first; fall back to a metadata file.
    if 'db_origin' in X_train.columns:
        db_origin_train = X_train['db_origin']
    elif 'db origin' in X_train.columns:
        db_origin_train = X_train['db origin']
    else:
        # Fall back: load from the complete training snapshot
        train_meta = pd.read_csv(ROOT + 'data/intermediate/train_data_complete.csv')
        # pick whichever column name exists
        col = 'db_origin' if 'db_origin' in train_meta.columns else 'db origin'
        db_origin_train = train_meta[col].reset_index(drop=True)

    db_origin_train = db_origin_train.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # --- treat NaN db_origin as its own origin group --- #
    # Rows with a missing origin are still valid training data; label them
    # so they participate in the incremental loop like every other origin.
    nan_label = config.get('incremental_training', {}).get('nan_origin_label', 'unknown')
    n_nan     = int(db_origin_train.isna().sum())
    if n_nan > 0:
        db_origin_train = db_origin_train.fillna(nan_label)
        logger.info(
            f"Filled {n_nan} NaN db_origin values with label '{nan_label}' "
            f"({n_nan / len(db_origin_train) * 100:.1f}% of training set)."
        )

    return X_train, y_train, X_test, y_test, db_origin_train


def get_origin_order(db_origin_series):
    """Return the order in which origins are added to the training set.

    Checks config['incremental_training']['origin_order'] first.
      - If it is a list, that list is used directly after validation.
      - If it is missing or None, falls back to largest-origin-first.

    Validation rules (applied only when a custom order is provided):
      - Every origin in the custom list must actually exist in the data.
        A typo here would silently skip an entire data source otherwise.
      - Every origin that exists in the data must appear in the custom list.
        Omitting one would silently drop training data with no warning.
    """
    available = set(db_origin_series.unique())

    custom_order = (
        config
        .get('incremental_training', {})
        .get('origin_order', None)
    )

    # --- no custom order → default: largest first --- #
    if custom_order is None:
        logger.info("No custom origin_order in config — using largest-first.")
        return db_origin_series.value_counts().index.tolist()

    # --- custom order provided → validate then use --- #
    nan_label = (
        config
        .get('incremental_training', {})
        .get('nan_origin_label', 'unknown')
    )
    specified = set(custom_order)

    unknown = specified - available
    missing = available - specified

    # If there are both unknown and missing entries, check whether any of
    # them are close matches — the most common cause is a typo in origin_order
    # or a mismatch between nan_origin_label and the origin_order entry.
    suggestions = {}
    if unknown and missing:
        for u in unknown:
            # simple character-overlap heuristic: flag pairs that share
            # more than 60 % of their longer length in common characters
            for m in missing:
                longer = max(len(u), len(m))
                common = len(set(u) & set(m))
                if longer > 0 and common / longer > 0.6:
                    suggestions[u] = m

    if unknown:
        hint_lines = []
        for u in sorted(unknown):
            if u in suggestions:
                hint_lines.append(f"  '{u}'  →  did you mean '{suggestions[u]}'?")
            else:
                hint_lines.append(f"  '{u}'")
        hints = "\n".join(hint_lines)

        raise ValueError(
            f"incremental_training.origin_order contains origins not found "
            f"in the training data:\n{hints}\n"
            f"Available origins: {sorted(available)}"
        )

    if missing:
        nan_note = ""
        if nan_label in missing:
            nan_note = (
                f"\n  Note: '{nan_label}' is the label assigned to rows where "
                f"db_origin was NaN (set via incremental_training.nan_origin_label). "
                f"Make sure origin_order uses exactly that string."
            )

        raise ValueError(
            f"incremental_training.origin_order is missing origins that "
            f"exist in the training data: {sorted(missing)}.\n"
            f"Every origin must be listed, or omit origin_order entirely "
            f"to use the default (largest-first) ordering."
            f"{nan_note}"
        )

    logger.info(f"Using custom origin_order from config: {custom_order}")
    return list(custom_order)


def load_all_hyperparams():
    """Load every optuna yaml once; keyed by model name."""
    hp = {}
    param_files = {
        'xgb':        'xgb_params.yaml',
        'logreg':     'logreg_params.yaml',
        'knn':        'knn_params.yaml',
        'naivebayes': 'naivebayes_params.yaml',
        'nn':         'nn_params.yaml',
    }
    for key, fname in param_files.items():
        path = ROOT + f'data/final/optuna_best_parameters/{fname}'
        if Path(path).exists():
            with open(path, 'r') as f:
                hp[key] = yaml.safe_load(f)
        else:
            logger.warning(f"Hyperparameter file not found: {path} — {key} will be skipped.")
    return hp


# ===========================================================================
# Imputation
# ===========================================================================

def impute_step(X_train_step, X_test_raw):
    method = config.get('imputation_method', {})

    if method.get('median') and method.get('mice'):
        raise ValueError("Both median and MICE are enabled — pick one.")

    # Identify and drop all-NaN columns (fitted on train only)
    all_nan_cols = X_train_step.columns[X_train_step.isna().all()].tolist()
    X_train_dropped = X_train_step.drop(columns=all_nan_cols)
    X_test_aligned = X_test_raw.reindex(columns=X_train_dropped.columns)

    if method.get('mice'):
        imputer = IterativeImputer(max_iter=1000)
    elif method.get('median'):
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    else:
        return X_train_step.copy(), X_test_raw.reindex(columns=X_train_step.columns).copy()

    imputer.fit(X_train_dropped)

    X_train_imp = pd.DataFrame(
        imputer.transform(X_train_dropped),
        columns=X_train_dropped.columns,
        index=X_train_step.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_aligned),
        columns=X_train_dropped.columns,
        index=X_test_raw.index,
    )

    # Restore all-NaN columns so downstream shape is consistent
    for col in all_nan_cols:
        X_train_imp[col] = np.nan
        X_test_imp[col] = np.nan

    # Restore original column order
    X_train_imp = X_train_imp[X_train_step.columns]
    X_test_imp = X_test_imp[X_train_step.columns]

    return X_train_imp, X_test_imp


# ===========================================================================
# SMOTE
# ===========================================================================

def apply_smote(X_train_imp, y_train):
    random_state = config.get('random_state', 42)
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train_imp, y_train)
    return pd.DataFrame(X_res, columns=X_train_imp.columns), pd.Series(y_res)


# ===========================================================================
# Training + evaluation for one step
# ===========================================================================

def train_and_evaluate_all(X_train_imp, X_train_raw_step, y_train,
                           X_test_imp, y_test, hyperparams):
    """Train every model, predict on X_test_imp, return {label: metrics}."""

    results = {}
    random_state = config.get('random_state', 42)

    # ---- XGBoost (imputed) ---- #
    if 'xgb' in hyperparams:
        logger.info("    Training XGBoost …")
        model = _train_xgb(X_train_imp, y_train, hyperparams['xgb'], random_state)
        results['xgboost'] = _evaluate(model, X_test_imp, y_test)

    # ---- XGBoost (unimputed — native missing support) ---- #
    logger.info("    Training XGBoost (unimputed) …")
    model = _train_xgb_unimputed(X_train_raw_step, y_train, random_state)
    results['xgboost_unimputed'] = _evaluate(model, X_test_imp if 'xgb' not in hyperparams
                                             else _drop_if_present(
                                                 pd.read_csv(ROOT + 'data/intermediate/X_test.csv').reset_index(drop=True),
                                                 'db_origin').pipe(_drop_if_present, 'db origin'),
                                             y_test, use_raw_test=True,
                                             X_test_raw=_drop_if_present(
                                                 _drop_if_present(
                                                     pd.read_csv(ROOT + 'data/intermediate/X_test.csv').reset_index(drop=True),
                                                     'db_origin'),
                                                 'db origin'))

    # ---- Logistic Regression ---- #
    if 'logreg' in hyperparams:
        logger.info("    Training Logistic Regression …")
        model = _train_logreg(X_train_imp, y_train, hyperparams['logreg'], random_state)
        results['logreg'] = _evaluate(model, X_test_imp, y_test)

    # ---- KNN ---- #
    if 'knn' in hyperparams:
        logger.info("    Training KNN …")
        model = _train_knn(X_train_imp, y_train, hyperparams['knn'])
        results['knn'] = _evaluate(model, X_test_imp, y_test)

    # ---- Naive Bayes ---- #
    if 'naivebayes' in hyperparams:
        logger.info("    Training Naive Bayes …")
        model = _train_naivebayes(X_train_imp, y_train, hyperparams['naivebayes'])
        results['naivebayes'] = _evaluate(model, X_test_imp, y_test)

    # ---- Neural Network ---- #
    if 'nn' in hyperparams:
        logger.info("    Training Neural Network …")
        model = _train_nn(X_train_imp, y_train, hyperparams['nn'])
        results['neural_network'] = _evaluate_nn(model, X_test_imp, y_test)

    return results


# ===========================================================================
# Individual model trainers  (mirror modeling.py logic, params passed in)
# ===========================================================================

def _train_xgb(X_train, y_train, params, random_state):
    params = {**params}                          # shallow copy — don't mutate
    params['n_jobs']       = -1
    params['tree_method']  = 'hist'
    return xgb.XGBClassifier(
        **params,
        objective='binary:logistic',
        enable_categorical=True,
        random_state=random_state,
        feature_names=list(X_train.columns),
    ).fit(X_train, y_train)


def _train_xgb_unimputed(X_train, y_train, random_state):
    """Hardcoded params matching modeling.py's train_xgb_unimputed."""
    return xgb.XGBClassifier(
        base_score=0.5, booster='gbtree',
        colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
        enable_categorical=False, gamma=0,
        learning_rate=0.1, max_delta_step=0, max_depth=2,
        min_child_weight=15, n_estimators=100, n_jobs=-1,
        num_parallel_tree=1, random_state=random_state,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
        tree_method='hist', verbosity=0,
    ).fit(X_train, y_train)


def _train_logreg(X_train, y_train, params, random_state):
    return LogisticRegression(
        **params, random_state=random_state, n_jobs=-1
    ).fit(X_train, y_train)


def _train_knn(X_train, y_train, params):
    return KNeighborsClassifier(
        **params, n_jobs=-1
    ).fit(X_train, y_train)


def _train_naivebayes(X_train, y_train, params):
    return GaussianNB(**params).fit(X_train, y_train)


def _train_nn(X_train, y_train, params):
    params = {**params}

    num_layers    = params.pop('num_layers')
    hidden_layers = tuple(params.pop(f'layer_{i}') for i in range(num_layers))
    max_epochs    = params.pop('epochs')
    params['max_epochs'] = max_epochs
    # remove any leftover layer_* keys that exceed num_layers
    params = {k: v for k, v in params.items() if not k.startswith('layer_')}

    torch.set_num_threads(32)

    X_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

    model = NeuralNetBinaryClassifier(
        PyTorchNN(input_dim=X_train.shape[1], hidden_layers=hidden_layers),
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        verbose=0,
        **params,
    )
    model.fit(X_tensor, y_tensor)
    return model


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def _evaluate(model, X_test, y_test, use_raw_test=False, X_test_raw=None):
    """Standard sklearn-compatible model evaluation."""
    if use_raw_test and X_test_raw is not None:
        X_eval = X_test_raw
    else:
        X_eval = X_test

    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    y_pred       = model.predict(X_eval)
    return _compute_metrics(y_test, y_pred_proba, y_pred)


def _evaluate_nn(model, X_test, y_test):
    """Neural network needs tensor input."""
    X_tensor     = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_pred_proba = model.predict_proba(X_tensor)[:, 1]
    y_pred       = model.predict(X_tensor)
    return _compute_metrics(y_test, y_pred_proba, y_pred)


def _compute_metrics(y_test, y_pred_proba, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    return {
        'auc':         float(auc(fpr, tpr)),
        'f1':          float(f1_score(y_test, y_pred)),
        'brier_score': float(brier_score_loss(y_test, y_pred_proba)),
    }


# ===========================================================================
# Tiny utility
# ===========================================================================

def _drop_if_present(df, col):
    return df.drop(columns=[col]) if col in df.columns else df


# ===========================================================================

if __name__ == "__main__":
    main()