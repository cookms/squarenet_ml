"""Repeated grouped-CV protocol and paired feature-ablation summaries."""
from __future__ import annotations

import hashlib
import inspect
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # Optional dependency; only required when model_name="xgboost".
    XGBClassifier = None
    XGBRegressor = None

from featurization import clean_text_scalar


LOWER_IS_BETTER = {"mae", "rmse", "log_loss", "brier"}
PRIMARY_METRIC = {"regression": "mae", "classification": "roc_auc"}

# A deliberately compact model panel:
# - linear: original regularized baseline
# - extra_trees: strong low-tuning nonlinear benchmark for tabular descriptors
# - random_forest: robust bagged-tree benchmark
# - hist_gradient_boosting: boosting model that often captures smoother interactions
DEFAULT_MODEL_NAMES = (
    "linear",
    "extra_trees",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
)


def regression_strata(y: pd.Series, max_bins: int = 10) -> pd.Series:
    values = pd.to_numeric(y, errors="coerce")
    bins = min(max_bins, max(2, len(values) // 20), values.nunique(dropna=True))
    if bins < 2:
        return pd.Series(np.zeros(len(values), dtype=int), index=y.index)
    return pd.qcut(
        values.rank(method="first"),
        q=int(bins),
        labels=False,
        duplicates="drop",
    ).astype(int)


def _valid_classification_split(y_train: pd.Series, y_test: pd.Series) -> bool:
    return y_train.nunique() >= 2 and y_test.nunique() >= 2


def generate_target_splits(
    data: pd.DataFrame,
    target: str,
    task: str,
    requested_splits: int,
    repeats: int,
    random_state: int,
    *,
    id_column: str = "material_id",
):
    cohort = data.loc[data[target].notna(), [id_column, "cv_group", target]].copy()
    cohort[target] = pd.to_numeric(cohort[target], errors="coerce")
    cohort = (
        cohort.dropna(subset=[target])
        .reset_index()
        .rename(columns={"index": "source_index"})
    )

    max_splits = min(requested_splits, cohort.cv_group.nunique())
    if task == "classification":
        class_counts = cohort[target].astype(int).value_counts()
        max_splits = (
            min(max_splits, int(class_counts.min()))
            if len(class_counts) >= 2
            else 1
        )
    if max_splits < 2:
        raise ValueError(f"{target}: fewer than two viable grouped folds.")

    strata = (
        cohort[target].astype(int)
        if task == "classification"
        else regression_strata(cohort[target])
    )

    chosen: list[dict[str, Any]] | None = None
    for n_splits in range(max_splits, 1, -1):
        candidate: list[dict[str, Any]] = []
        valid = True

        for repeat in range(repeats):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state + repeat,
            )
            try:
                repeat_splits = list(
                    splitter.split(cohort, strata, groups=cohort["cv_group"])
                )
            except ValueError:
                valid = False
                break

            for fold, (train_pos, test_pos) in enumerate(repeat_splits):
                train_groups = set(cohort.iloc[train_pos]["cv_group"])
                test_groups = set(cohort.iloc[test_pos]["cv_group"])

                if train_groups & test_groups:
                    raise AssertionError(
                        "Group leakage detected while constructing CV splits."
                    )

                if (
                    task == "classification"
                    and not _valid_classification_split(
                        cohort.iloc[train_pos][target],
                        cohort.iloc[test_pos][target],
                    )
                ):
                    valid = False
                    break

                candidate.append(
                    {
                        "target": target,
                        "repeat": repeat,
                        "fold": fold,
                        "split_id": f"{target}__r{repeat:02d}_f{fold:02d}",
                        "train_source_index": cohort.iloc[train_pos][
                            "source_index"
                        ].to_numpy(),
                        "test_source_index": cohort.iloc[test_pos][
                            "source_index"
                        ].to_numpy(),
                        "n_train": len(train_pos),
                        "n_test": len(test_pos),
                        "n_train_groups": len(train_groups),
                        "n_test_groups": len(test_groups),
                    }
                )

            if not valid:
                break

        if valid:
            chosen = candidate
            break

    if chosen is None:
        raise ValueError(
            f"{target}: could not construct grouped folds with valid train/test targets."
        )

    manifest = []
    for split in chosen:
        for material_id, group_id in zip(
            data.loc[split["test_source_index"], id_column],
            data.loc[split["test_source_index"], "cv_group"],
        ):
            manifest.append(
                {
                    "target": target,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "split_id": split["split_id"],
                    "material_id": material_id,
                    "cv_group": group_id,
                    "role": "test",
                }
            )

    return pd.DataFrame(manifest), chosen


def sklearn_safe_frame(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    for column in result.columns:
        series = result[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            result[column] = pd.to_numeric(series, errors="coerce").astype("float64")
        else:
            result[column] = pd.Series(
                [
                    np.nan
                    if clean_text_scalar(v) is None
                    else clean_text_scalar(v)
                    for v in series.tolist()
                ],
                index=result.index,
                dtype=object,
            )
    return result


def _compatible_one_hot(
    min_frequency: int,
    *,
    sparse_output: bool = True,
) -> OneHotEncoder:
    kwargs: dict[str, Any] = {
        "handle_unknown": "ignore",
        "min_frequency": min_frequency,
    }
    parameter = (
        "sparse_output"
        if "sparse_output" in inspect.signature(OneHotEncoder).parameters
        else "sparse"
    )
    kwargs[parameter] = sparse_output
    return OneHotEncoder(**kwargs)


def make_preprocessor(
    X: pd.DataFrame,
    min_frequency: int,
    *,
    dense_output: bool = False,
) -> ColumnTransformer:
    numeric = [
        c for c in X
        if pd.api.types.is_numeric_dtype(X[c])
        or pd.api.types.is_bool_dtype(X[c])
    ]
    categorical = [c for c in X if c not in numeric]

    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                )),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                )),
                ("onehot", _compatible_one_hot(
                    min_frequency,
                    sparse_output=not dense_output,
                )),
            ]),
            categorical,
        ))

    if not transformers:
        raise ValueError("No usable columns were supplied to the preprocessor.")

    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0 if dense_output else 1.0,
        verbose_feature_names_out=False,
    )


def make_model(
    task: str,
    X: pd.DataFrame,
    config: Any,
    min_frequency: int,
    model_name: str = "linear",
    y_train: pd.Series | None = None,
) -> Pipeline:
    """Construct one model using the same fold-local preprocessing.

    Supported models
    ----------------
    linear
        Ridge regression or balanced logistic regression. This is the original
        baseline and is retained for direct comparison.
    extra_trees
        Extremely randomized trees. Usually a strong nonlinear benchmark for
        medium-sized tabular descriptor sets and relatively insensitive to
        feature scaling.
    random_forest
        Bagged random forest. More conservative than ExtraTrees and useful as a
        second nonlinear reference.
    hist_gradient_boosting
        Histogram gradient boosting. Uses dense transformed features and can
        capture smoother higher-order interactions.
    xgboost
        Gradient-boosted decision trees via XGBoost. Supports sparse one-hot
        encoded inputs and is often a strong performer on tabular descriptor data.

    Hyperparameters can optionally be overridden by adding attributes with the
    names used below to ExperimentConfig; otherwise sensible defaults are used.
    """
    model_name = str(model_name).lower().strip()
    valid = set(DEFAULT_MODEL_NAMES)
    if model_name not in valid:
        raise ValueError(
            f"Unknown model_name={model_name!r}. "
            f"Choose one of {sorted(valid)}."
        )

    dense_output = model_name == "hist_gradient_boosting"
    preprocess = make_preprocessor(
        X,
        min_frequency,
        dense_output=dense_output,
    )

    random_state = int(getattr(config, "random_state", 42))

    if task == "regression":
        if model_name == "linear":
            estimator = TransformedTargetRegressor(
                regressor=Ridge(
                    alpha=float(getattr(config, "ridge_alpha", 10.0)),
                    solver="lsqr",
                ),
                transformer=StandardScaler(),
            )

        elif model_name == "extra_trees":
            estimator = ExtraTreesRegressor(
                n_estimators=int(getattr(config, "extra_trees_n_estimators", 500)),
                max_features=getattr(config, "extra_trees_max_features", 1.0),
                min_samples_leaf=int(
                    getattr(config, "extra_trees_min_samples_leaf", 1)
                ),
                n_jobs=int(getattr(config, "n_jobs", -1)),
                random_state=random_state,
            )

        elif model_name == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=int(
                    getattr(config, "random_forest_n_estimators", 500)
                ),
                max_features=getattr(config, "random_forest_max_features", 0.7),
                min_samples_leaf=int(
                    getattr(config, "random_forest_min_samples_leaf", 1)
                ),
                n_jobs=int(getattr(config, "n_jobs", -1)),
                random_state=random_state,
            )

        elif model_name == "hist_gradient_boosting":
            estimator = HistGradientBoostingRegressor(
                learning_rate=float(
                    getattr(config, "hist_gradient_boosting_learning_rate", 0.05)
                ),
                max_iter=int(
                    getattr(config, "hist_gradient_boosting_max_iter", 300)
                ),
                max_leaf_nodes=int(
                    getattr(config, "hist_gradient_boosting_max_leaf_nodes", 31)
                ),
                l2_regularization=float(
                    getattr(config, "hist_gradient_boosting_l2", 1.0)
                ),
                early_stopping=True,
                random_state=random_state,
            )

        else:  # xgboost
            if XGBRegressor is None:
                raise ImportError(
                    'xgboost is required for model_name="xgboost". '
                    'Install it with `pip install xgboost`.'
                )
            estimator = XGBRegressor(
                n_estimators=int(getattr(config, "xgb_n_estimators", 500)),
                learning_rate=float(getattr(config, "xgb_learning_rate", 0.05)),
                max_depth=int(getattr(config, "xgb_max_depth", 6)),
                min_child_weight=float(getattr(config, "xgb_min_child_weight", 1.0)),
                subsample=float(getattr(config, "xgb_subsample", 0.8)),
                colsample_bytree=float(getattr(config, "xgb_colsample_bytree", 0.8)),
                reg_alpha=float(getattr(config, "xgb_reg_alpha", 0.0)),
                reg_lambda=float(getattr(config, "xgb_reg_lambda", 1.0)),
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=int(getattr(config, "n_jobs", -1)),
                random_state=random_state,
            )

    elif task == "classification":
        if model_name == "linear":
            estimator = LogisticRegression(
                C=float(getattr(config, "logistic_c", 1.0)),
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=random_state,
            )

        elif model_name == "extra_trees":
            estimator = ExtraTreesClassifier(
                n_estimators=int(getattr(config, "extra_trees_n_estimators", 500)),
                max_features=getattr(config, "extra_trees_max_features", "sqrt"),
                min_samples_leaf=int(
                    getattr(config, "extra_trees_min_samples_leaf", 1)
                ),
                class_weight="balanced",
                n_jobs=int(getattr(config, "n_jobs", -1)),
                random_state=random_state,
            )

        elif model_name == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=int(
                    getattr(config, "random_forest_n_estimators", 500)
                ),
                max_features=getattr(config, "random_forest_max_features", "sqrt"),
                min_samples_leaf=int(
                    getattr(config, "random_forest_min_samples_leaf", 1)
                ),
                class_weight="balanced_subsample",
                n_jobs=int(getattr(config, "n_jobs", -1)),
                random_state=random_state,
            )

        elif model_name == "hist_gradient_boosting":
            estimator = HistGradientBoostingClassifier(
                learning_rate=float(
                    getattr(config, "hist_gradient_boosting_learning_rate", 0.05)
                ),
                max_iter=int(
                    getattr(config, "hist_gradient_boosting_max_iter", 300)
                ),
                max_leaf_nodes=int(
                    getattr(config, "hist_gradient_boosting_max_leaf_nodes", 31)
                ),
                l2_regularization=float(
                    getattr(config, "hist_gradient_boosting_l2", 1.0)
                ),
                early_stopping=True,
                random_state=random_state,
            )

        else:  # xgboost
            if XGBClassifier is None:
                raise ImportError(
                    'xgboost is required for model_name="xgboost". '
                    'Install it with `pip install xgboost`.'
                )

            scale_pos_weight = getattr(config, "xgb_scale_pos_weight", None)
            if scale_pos_weight is None and y_train is not None:
                y_numeric = pd.to_numeric(y_train, errors="coerce").dropna().astype(int)
                positives = int((y_numeric == 1).sum())
                negatives = int((y_numeric == 0).sum())
                scale_pos_weight = negatives / positives if positives > 0 else 1.0
            if scale_pos_weight is None:
                scale_pos_weight = 1.0

            estimator = XGBClassifier(
                n_estimators=int(getattr(config, "xgb_n_estimators", 500)),
                learning_rate=float(getattr(config, "xgb_learning_rate", 0.05)),
                max_depth=int(getattr(config, "xgb_max_depth", 6)),
                min_child_weight=float(getattr(config, "xgb_min_child_weight", 1.0)),
                subsample=float(getattr(config, "xgb_subsample", 0.8)),
                colsample_bytree=float(getattr(config, "xgb_colsample_bytree", 0.8)),
                reg_alpha=float(getattr(config, "xgb_reg_alpha", 0.0)),
                reg_lambda=float(getattr(config, "xgb_reg_lambda", 1.0)),
                scale_pos_weight=float(scale_pos_weight),
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=int(getattr(config, "n_jobs", -1)),
                random_state=random_state,
            )
    else:
        raise ValueError(
            f"Unknown task={task!r}; expected 'regression' or 'classification'."
        )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    both = np.unique(y_true).size == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both else np.nan,
        "average_precision": (
            float(average_precision_score(y_true, y_score))
            if both
            else np.nan
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_feature_sets(
    data: pd.DataFrame,
    target_registry: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: Any,
    *,
    model_names: Sequence[str] | None = None,
    id_column: str = "material_id",
    formula_column: str = "formula",
):
    """Evaluate feature sets and algorithms on identical grouped splits.

    Parameters
    ----------
    model_names
        Sequence of model identifiers accepted by ``make_model``. If omitted,
        all models in ``DEFAULT_MODEL_NAMES`` are evaluated.

    Notes
    -----
    Every model/feature-set pair sees the exact same target-specific grouped
    train/test indices, so fold-wise comparisons remain paired.
    """
    if model_names is None:
        model_names = DEFAULT_MODEL_NAMES
    model_names = tuple(dict.fromkeys(str(name) for name in model_names))

    scores: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    manifests: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []

    for _, spec in target_registry.iterrows():
        target, task = spec["target"], spec["task"]
        print(f"\n{spec.display_name}: building shared grouped splits...")

        try:
            manifest, splits = generate_target_splits(
                data,
                target,
                task,
                config.requested_splits,
                config.requested_repeats,
                config.random_state,
                id_column=id_column,
            )
        except Exception as exc:
            failures.append(
                {
                    "target": target,
                    "stage": "split generation",
                    "error": repr(exc),
                }
            )
            print(f"  skipped: {exc}")
            continue

        manifests.append(manifest)
        print(
            f"  {len(splits)} paired splits "
            f"({config.requested_repeats} repeats x "
            f"{len(splits) // config.requested_repeats} folds)"
        )

        for feature_set_name, columns in feature_sets.items():
            print(
                f"  feature set: {feature_set_name} "
                f"({len(columns)} raw columns)"
            )

            for model_name in model_names:
                print(f"    model: {model_name}")

                for split in splits:
                    train_index = split["train_source_index"]
                    test_index = split["test_source_index"]

                    X_train = sklearn_safe_frame(
                        data.loc[train_index, columns]
                    )
                    X_test = sklearn_safe_frame(
                        data.loc[test_index, columns]
                    )
                    y_train = pd.to_numeric(
                        data.loc[train_index, target],
                        errors="coerce",
                    )
                    y_test = pd.to_numeric(
                        data.loc[test_index, target],
                        errors="coerce",
                    )

                    try:
                        model = make_model(
                            task,
                            X_train,
                            config,
                            config.onehot_min_frequency,
                            model_name=model_name,
                            y_train=y_train,
                        )
                        model.fit(X_train, y_train)

                        if task == "regression":
                            y_pred = np.asarray(
                                model.predict(X_test),
                                dtype=float,
                            )
                            metric_values = _regression_metrics(
                                y_test.to_numpy(dtype=float),
                                y_pred,
                            )
                            y_score = np.full(len(y_pred), np.nan)

                        else:
                            y_score = np.asarray(
                                model.predict_proba(X_test)[:, 1],
                                dtype=float,
                            )
                            y_pred = (y_score >= 0.5).astype(int)
                            metric_values = _classification_metrics(
                                y_test.to_numpy(dtype=int),
                                y_score,
                            )

                        try:
                            transformed = len(
                                model.named_steps[
                                    "preprocess"
                                ].get_feature_names_out()
                            )
                        except Exception:
                            transformed = np.nan

                        common_score_fields = {
                            "target": target,
                            "target_name": spec.display_name,
                            "task": task,
                            "feature_set": feature_set_name,
                            "model_name": model_name,
                            "repeat": split["repeat"],
                            "fold": split["fold"],
                            "split_id": split["split_id"],
                            "n_train": split["n_train"],
                            "n_test": split["n_test"],
                            "n_train_groups": split["n_train_groups"],
                            "n_test_groups": split["n_test_groups"],
                            "n_raw_features": len(columns),
                            "n_transformed_features": transformed,
                        }

                        for metric, score in metric_values.items():
                            scores.append(
                                {
                                    **common_score_fields,
                                    "metric": metric,
                                    "score": score,
                                }
                            )

                        for pos, source_index in enumerate(test_index):
                            predictions.append(
                                {
                                    "target": target,
                                    "target_name": spec.display_name,
                                    "task": task,
                                    "feature_set": feature_set_name,
                                    "model_name": model_name,
                                    "repeat": split["repeat"],
                                    "fold": split["fold"],
                                    "split_id": split["split_id"],
                                    "source_index": int(source_index),
                                    "material_id": data.loc[
                                        source_index, id_column
                                    ],
                                    "cv_group": data.loc[
                                        source_index, "cv_group"
                                    ],
                                    "formula": data.loc[
                                        source_index, formula_column
                                    ],
                                    "y_true": float(y_test.iloc[pos]),
                                    "y_pred": float(y_pred[pos]),
                                    "y_score": (
                                        float(y_score[pos])
                                        if np.isfinite(y_score[pos])
                                        else np.nan
                                    ),
                                }
                            )

                    except Exception as exc:
                        failures.append(
                            {
                                "target": target,
                                "feature_set": feature_set_name,
                                "model_name": model_name,
                                "split_id": split["split_id"],
                                "stage": "fit/evaluate",
                                "error": repr(exc),
                            }
                        )

    return (
        pd.DataFrame(scores),
        pd.DataFrame(predictions),
        (
            pd.concat(manifests, ignore_index=True)
            if manifests
            else pd.DataFrame()
        ),
        pd.DataFrame(failures),
        splits,
    )


def stable_seed(*parts: Any, base: int = 0) -> int:
    payload = "||".join(map(str, parts)).encode("utf-8")
    return (
        base
        + int.from_bytes(
            hashlib.sha256(payload).digest()[:4],
            "big",
        )
        % 100000
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    draws: int,
    seed: int,
    confidence: float = 0.95,
):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        return np.nan, np.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])

    sampled = (
        np.random.default_rng(seed)
        .choice(
            array,
            size=(draws, len(array)),
            replace=True,
        )
        .mean(axis=1)
    )
    alpha = (1 - confidence) / 2
    return tuple(
        np.quantile(
            sampled,
            [alpha, 1 - alpha],
        ).astype(float)
    )


def _ensure_model_name(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep old single-model score tables compatible with new helpers."""
    if "model_name" in frame.columns:
        return frame
    result = frame.copy()
    result["model_name"] = "linear"
    return result


def summarize_scores(scores: pd.DataFrame, config: Any) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    scores = _ensure_model_name(scores)
    group_cols = [
        "target",
        "target_name",
        "task",
        "feature_set",
        "model_name",
        "metric",
    ]

    repeat_scores = (
        scores.groupby(group_cols + ["repeat"], as_index=False)
        .agg(repeat_mean_score=("score", "mean"))
    )

    rows = []
    for key, group in repeat_scores.groupby(group_cols, sort=False):
        (
            target,
            target_name,
            task,
            feature_set,
            model_name,
            metric,
        ) = key

        ci_low, ci_high = bootstrap_mean_ci(
            group.repeat_mean_score,
            config.bootstrap_draws,
            stable_seed(
                "score",
                target,
                feature_set,
                model_name,
                metric,
                base=config.random_state,
            ),
        )

        fold_group = scores[
            (scores.target == target)
            & (scores.feature_set == feature_set)
            & (scores.model_name == model_name)
            & (scores.metric == metric)
        ]

        rows.append(
            {
                "target": target,
                "target_name": target_name,
                "task": task,
                "feature_set": feature_set,
                "model_name": model_name,
                "metric": metric,
                "mean_score": float(
                    group.repeat_mean_score.mean()
                ),
                "repeat_std": (
                    float(
                        group.repeat_mean_score.std(ddof=1)
                    )
                    if len(group) > 1
                    else np.nan
                ),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_repeats": int(group.repeat.nunique()),
                "n_fold_scores": int(
                    fold_group.score.notna().sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def paired_against_baseline(
    scores: pd.DataFrame,
    baseline: str = "Baseline",
) -> pd.DataFrame:
    """Compare feature sets against Baseline within the same algorithm."""
    if scores.empty:
        return pd.DataFrame()

    scores = _ensure_model_name(scores)

    keys = [
        "target",
        "target_name",
        "task",
        "model_name",
        "repeat",
        "fold",
        "split_id",
        "metric",
    ]

    reference = (
        scores.loc[
            scores.feature_set == baseline,
            keys + ["score"],
        ]
        .rename(columns={"score": "baseline_score"})
    )

    candidates = (
        scores.loc[
            scores.feature_set != baseline,
            keys + ["feature_set", "score"],
        ]
        .rename(columns={"score": "candidate_score"})
    )

    paired = candidates.merge(
        reference,
        on=keys,
        how="inner",
        validate="many_to_one",
    )

    direction = np.where(
        paired.metric.isin(LOWER_IS_BETTER),
        -1.0,
        1.0,
    )
    paired["improvement"] = direction * (
        paired["candidate_score"]
        - paired["baseline_score"]
    )
    paired["improved"] = paired["improvement"] > 0
    return paired


def summarize_paired(
    paired: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()

    paired = _ensure_model_name(paired)

    group_cols = [
        "target",
        "target_name",
        "task",
        "feature_set",
        "model_name",
        "metric",
    ]

    repeat_delta = (
        paired.groupby(group_cols + ["repeat"], as_index=False)
        .agg(
            repeat_mean_improvement=("improvement", "mean")
        )
    )

    rows = []
    for key, group in repeat_delta.groupby(
        group_cols,
        sort=False,
    ):
        (
            target,
            target_name,
            task,
            feature_set,
            model_name,
            metric,
        ) = key

        ci_low, ci_high = bootstrap_mean_ci(
            group.repeat_mean_improvement,
            config.bootstrap_draws,
            stable_seed(
                "paired",
                target,
                feature_set,
                model_name,
                metric,
                base=config.random_state,
            ),
        )

        fold_group = paired[
            (paired.target == target)
            & (paired.feature_set == feature_set)
            & (paired.model_name == model_name)
            & (paired.metric == metric)
        ]

        win_rate = float(fold_group.improved.mean())

        evidence = (
            "consistent improvement"
            if ci_low > 0 and win_rate >= 0.60
            else (
                "consistent degradation"
                if ci_high < 0
                else "inconclusive"
            )
        )

        rows.append(
            {
                "target": target,
                "target_name": target_name,
                "task": task,
                "feature_set": feature_set,
                "model_name": model_name,
                "metric": metric,
                "mean_improvement": float(
                    group.repeat_mean_improvement.mean()
                ),
                "median_fold_improvement": float(
                    fold_group.improvement.median()
                ),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "fold_win_rate": win_rate,
                "n_paired_folds": int(
                    fold_group.improvement.notna().sum()
                ),
                "n_repeats": int(
                    group.repeat.nunique()
                ),
                "evidence": evidence,
            }
        )

    return pd.DataFrame(rows)


def primary_metric_tables(
    score_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    target_registry: pd.DataFrame,
):
    score_rows, paired_rows = [], []

    for _, spec in target_registry.iterrows():
        metric = PRIMARY_METRIC[spec.task]

        subset = score_summary[
            (score_summary.target == spec.target)
            & (score_summary.metric == metric)
        ]
        paired_subset = paired_summary[
            (paired_summary.target == spec.target)
            & (paired_summary.metric == metric)
        ]

        if subset.empty and spec.task == "classification":
            metric = "balanced_accuracy"
            subset = score_summary[
                (score_summary.target == spec.target)
                & (score_summary.metric == metric)
            ]
            paired_subset = paired_summary[
                (paired_summary.target == spec.target)
                & (paired_summary.metric == metric)
            ]

        score_rows.append(subset)
        paired_rows.append(paired_subset)

    return (
        (
            pd.concat(score_rows, ignore_index=True)
            if score_rows
            else pd.DataFrame()
        ),
        (
            pd.concat(paired_rows, ignore_index=True)
            if paired_rows
            else pd.DataFrame()
        ),
    )
