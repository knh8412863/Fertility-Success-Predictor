# Run after notebooks/02_preprocessing_final.ipynb has created data/X_train.csv, data/y_train.csv, data/X_test.csv, and data/test_ids.csv.

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS = 5
SEEDS = [42, 2025, 7, 123, 777]
TARGET = "임신 성공 여부"

# 심사용 기본값은 False. True로 바꾸면 같은 이름의 캐시가 있을 때 재사용합니다.
USE_CACHE = False

root = Path.cwd()
if not (root / "data").exists() and (root.parent / "data").exists():
    root = root.parent

DATA_DIR = root / "data"
SUB_DIR = root / "submission"
SUB_DIR.mkdir(exist_ok=True)

CACHE_DAYLONG = SUB_DIR / "oof_cache_daylong"
CACHE_FINAL = SUB_DIR / "oof_cache_final"
CACHE_RAW = SUB_DIR / "oof_cache_raw_catboost"
CACHE_FEATURE = SUB_DIR / "oof_cache_feature_count_sweep"
for path in [CACHE_DAYLONG, CACHE_FINAL, CACHE_RAW, CACHE_FEATURE]:
    path.mkdir(exist_ok=True)

print("root:", root)
print("use_cache:", USE_CACHE)

X_all = pd.read_csv(DATA_DIR / "X_train.csv")
Xt_all = pd.read_csv(DATA_DIR / "X_test.csv")
y = pd.read_csv(DATA_DIR / "y_train.csv").squeeze().astype(int)
test_ids = pd.read_csv(DATA_DIR / "test_ids.csv").squeeze()
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

assert len(X_all) == len(y)
assert len(Xt_all) == len(test_ids)
assert list(test_ids) == list(sample_sub["ID"])

SPW = (y == 0).sum() / (y == 1).sum()
print("X:", X_all.shape, "Xt:", Xt_all.shape, "target_mean:", y.mean(), "spw:", SPW)

BASE_TE_COLS = ["시술 시기 코드", "특정 시술 유형", "배아 생성 주요 이유", "난자 출처", "정자 출처"]
BASE_FREQ_COLS = ["시술 시기 코드", "특정 시술 유형", "배아 생성 주요 이유", "시술 유형"]


def smooth_target_map(col_values, target, smooth=80):
    global_mean = target.mean()
    stats = target.groupby(col_values).agg(["mean", "count"])
    mapping = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)
    return mapping, global_mean


def add_fold_features(X_tr, y_tr, X_val, X_te, seed, smooth=80):
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()
    te_cols = [c for c in BASE_TE_COLS if c in X_tr.columns]
    freq_cols = [c for c in BASE_FREQ_COLS if c in X_tr.columns]

    for col in freq_cols:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[f"{col}_freq"] = X_tr[col].map(freq).fillna(0)
        X_val[f"{col}_freq"] = X_val[col].map(freq).fillna(0)
        X_te[f"{col}_freq"] = X_te[col].map(freq).fillna(0)

    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for col in te_cols:
        oof_te = pd.Series(index=X_tr.index, dtype=float)
        for in_idx, out_idx in inner.split(X_tr, y_tr):
            in_rows = X_tr.index[in_idx]
            out_rows = X_tr.index[out_idx]
            mapping, global_mean = smooth_target_map(X_tr.loc[in_rows, col], y_tr.loc[in_rows], smooth=smooth)
            oof_te.loc[out_rows] = X_tr.loc[out_rows, col].map(mapping).fillna(global_mean)
        mapping, global_mean = smooth_target_map(X_tr[col], y_tr, smooth=smooth)
        X_tr[f"{col}_te"] = oof_te.fillna(global_mean)
        X_val[f"{col}_te"] = X_val[col].map(mapping).fillna(global_mean)
        X_te[f"{col}_te"] = X_te[col].map(mapping).fillna(global_mean)
    return X_tr, X_val, X_te


def save_pair(cache_dir, name, oof, test):
    np.save(cache_dir / f"{name}_oof.npy", oof)
    np.save(cache_dir / f"{name}_test.npy", test)


def maybe_load_pair(cache_dir, name):
    oof_path = cache_dir / f"{name}_oof.npy"
    test_path = cache_dir / f"{name}_test.npy"
    if USE_CACHE and oof_path.exists() and test_path.exists():
        oof = np.load(oof_path)
        test = np.load(test_path)
        print("loaded", name, roc_auc_score(y, oof))
        return oof, test
    return None


def rank01(a):
    return pd.Series(a).rank(method="average").to_numpy() / (len(a) + 1)

RANK_PARAMS = {
    "n_estimators": 2500,
    "learning_rate": 0.02149524140540219,
    "num_leaves": 16,
    "max_depth": 13,
    "min_child_samples": 27,
    "subsample": 0.6808936174607194,
    "subsample_freq": 1,
    "colsample_bytree": 0.5140506756742775,
    "reg_alpha": 9.081327528794018,
    "reg_lambda": 8.9592839180652,
    "objective": "binary",
    "metric": "auc",
    "scale_pos_weight": SPW,
    "n_jobs": -1,
    "verbose": -1,
}

COMPACT_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.018,
    "num_leaves": 48,
    "max_depth": 6,
    "min_child_samples": 110,
    "subsample": 0.75,
    "subsample_freq": 1,
    "colsample_bytree": 0.55,
    "reg_alpha": 8.0,
    "reg_lambda": 8.0,
    "objective": "binary",
    "metric": "auc",
    "scale_pos_weight": SPW,
    "n_jobs": -1,
    "verbose": -1,
}

LGBM72_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.020,
    "num_leaves": 24,
    "max_depth": 10,
    "min_child_samples": 35,
    "subsample": 0.72,
    "subsample_freq": 1,
    "colsample_bytree": 0.58,
    "reg_alpha": 5.0,
    "reg_lambda": 7.0,
    "objective": "binary",
    "metric": "auc",
    "scale_pos_weight": SPW,
    "n_jobs": -1,
    "verbose": -1,
}

REVIEW_EXACT_PARAMS = {
    "n_estimators": 2500,
    "learning_rate": 0.02149524140540219,
    "num_leaves": 16,
    "max_depth": 13,
    "min_child_samples": 27,
    "subsample": 0.6808936174607194,
    "subsample_freq": 1,
    "colsample_bytree": 0.5140506756742775,
    "reg_alpha": 9.081327528794018,
    "reg_lambda": 8.9592839180652,
    "objective": "binary",
    "metric": "auc",
    "scale_pos_weight": SPW,
    "n_jobs": -1,
    "verbose": -1,
}

DROP_BEST = [
    "배란유도_기록됨", "현재시술_목적", "저반응_여부", "고반응_여부",
    "is_mix_date_missing", "난자 해동 경과일",
    "불임 원인 - 정자 형태", "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 농도",
    "불임 원인 - 자궁경부 문제", "배아 해동 경과일_결측", "신선 배아 사용 여부",
    "저장된 신선 난자 수", "불임 원인 - 정자 운동성", "초고령_43이상", "대리모 여부",
    "난자 혼합 경과일_결측", "IVF_성공이력", "여성 부 불임 원인",
    "동결 배아 사용 여부", "잉여배아_존재", "장기배양_이식",
]
DROP_COMPACT_EXTRA = [
    "남성 주 불임 원인", "여성 주 불임 원인", "부부 주 불임 원인",
    "기증 배아 사용 여부", "배아 해동 경과일", "난자 해동 경과일_결측",
    "is_eset", "is_blastocyst", "is_transfer_canceled",
]
COMPACT_DROPS = [c for c in DROP_BEST + DROP_COMPACT_EXTRA if c in X_all.columns]


def cv_lgbm(seed, X_base, Xt_base, params, selected_features=None, name="lgbm", early_stop=200):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_base))
    test_pred = np.zeros(len(Xt_base))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_base, y), start=1):
        X_tr = X_base.iloc[tr_idx].copy()
        X_val = X_base.iloc[val_idx].copy()
        X_te = Xt_base.copy()
        y_tr = y.iloc[tr_idx].copy()
        y_val = y.iloc[val_idx].copy()
        X_tr, X_val, X_te = add_fold_features(X_tr, y_tr, X_val, X_te, seed=seed + fold, smooth=80)

        if selected_features is not None:
            cols = [c for c in selected_features if c in X_tr.columns]
            X_tr = X_tr[cols]
            X_val = X_val[cols]
            X_te = X_te[cols]
        else:
            cols = X_tr.columns

        model = LGBMClassifier(**params, random_state=seed + fold)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(early_stop, verbose=False), log_evaluation(-1)],
        )
        pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = pred
        test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        print(f"{name} seed={seed} fold={fold}: AUC={roc_auc_score(y_val, pred):.6f}, iter={model.best_iteration_}, cols={len(cols)}")

    print(f"{name} seed={seed} OOF={roc_auc_score(y, oof):.6f}")
    return oof, test_pred


def get_compact():
    loaded = maybe_load_pair(CACHE_DAYLONG, "compact_regularized_enc")
    if loaded is not None:
        return loaded
    X = X_all.drop(columns=COMPACT_DROPS)
    Xt = Xt_all.drop(columns=COMPACT_DROPS)
    oofs, tests = [], []
    for seed in SEEDS:
        oof, test = cv_lgbm(seed, X, Xt, COMPACT_PARAMS, name="compact", early_stop=250)
        oofs.append(oof)
        tests.append(test)
    oof = np.mean(oofs, axis=0)
    test = np.mean(tests, axis=0)
    save_pair(CACHE_DAYLONG, "compact_regularized_enc", oof, test)
    print("compact multi-seed:", roc_auc_score(y, oof))
    return oof, test

def build_importance():
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    importances = []
    rank_oof = np.zeros(len(X_all))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y), start=1):
        X_tr = X_all.iloc[tr_idx].copy()
        X_val = X_all.iloc[val_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        y_val = y.iloc[val_idx].copy()
        X_dummy = X_val.iloc[:1].copy()
        X_tr, X_val, _ = add_fold_features(X_tr, y_tr, X_val, X_dummy, seed=RANDOM_STATE + fold, smooth=80)

        model = LGBMClassifier(**RANK_PARAMS, random_state=RANDOM_STATE + fold)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(200, verbose=False), log_evaluation(-1)],
        )
        pred = model.predict_proba(X_val)[:, 1]
        rank_oof[val_idx] = pred
        importances.append(pd.DataFrame({"feature": X_tr.columns, "importance": model.feature_importances_}))
        print(f"rank fold={fold}: AUC={roc_auc_score(y_val, pred):.6f}, iter={model.best_iteration_}")

    imp = pd.concat(importances).groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    imp.to_csv(SUB_DIR / f"lgbm_final_importance_{ts}.csv", index=False)
    print("rank OOF:", roc_auc_score(y, rank_oof), "features:", len(imp))
    return imp


importance_df = build_importance()
feature_rank = importance_df["feature"].tolist()
selected_72 = feature_rank[:72]
selected_80 = feature_rank[:80]


def get_lgbm72():
    loaded = maybe_load_pair(CACHE_FINAL, "lgbm72")
    if loaded is not None:
        return loaded
    oofs, tests = [], []
    for seed in SEEDS:
        oof, test = cv_lgbm(seed, X_all, Xt_all, LGBM72_PARAMS, selected_features=selected_72, name="lgbm72", early_stop=200)
        oofs.append(oof)
        tests.append(test)
    oof = np.mean(oofs, axis=0)
    test = np.mean(tests, axis=0)
    save_pair(CACHE_FINAL, "lgbm72", oof, test)
    print("lgbm72 multi-seed:", roc_auc_score(y, oof))
    return oof, test


def get_feature_count_nf80():
    loaded = maybe_load_pair(CACHE_FEATURE, "review_exact_nf80_5seed")
    if loaded is not None:
        return loaded
    oofs, tests = [], []
    for seed in SEEDS:
        oof, test = cv_lgbm(seed, X_all, Xt_all, REVIEW_EXACT_PARAMS, selected_features=selected_80, name="review_exact_nf80", early_stop=200)
        oofs.append(oof)
        tests.append(test)
    oof = np.mean(oofs, axis=0)
    test = np.mean(tests, axis=0)
    save_pair(CACHE_FEATURE, "review_exact_nf80_5seed", oof, test)
    print("review_exact_nf80 multi-seed:", roc_auc_score(y, oof))
    return oof, test

COUNT_MAP = {"0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6}
AGE_MAP = {"만18-34세": 0, "만35-37세": 1, "만38-39세": 2, "만40-42세": 3, "만43-44세": 4, "만45-50세": 5, "알 수 없음": np.nan}
DONOR_EGG_AGE_MAP = {"알 수 없음": np.nan, "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2, "만31-35세": 3}
DONOR_SPERM_AGE_MAP = {"알 수 없음": np.nan, "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2, "만31-35세": 3, "만36-40세": 4, "만41-45세": 5}
DROP_RAW_COLS = ["불임 원인 - 여성 요인", "난자 채취 경과일"]
DI_ZERO_COLS = [
    "미세주입된 난자 수", "미세주입에서 생성된 배아 수", "총 생성 배아 수", "이식된 배아 수",
    "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수",
    "해동 난자 수", "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수", "난자 혼합 경과일",
    "배아 이식 경과일", "배아 해동 경과일", "동결 배아 사용 여부", "신선 배아 사용 여부",
    "기증 배아 사용 여부", "단일 배아 이식 여부", "임신 시도 또는 마지막 임신 경과 연수",
]


def safe_num(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def preprocess_raw(df):
    df = df.copy().drop(columns=[c for c in DROP_RAW_COLS if c in df.columns], errors="ignore")

    for col in ["배아 이식 경과일", "난자 혼합 경과일", "배아 해동 경과일", "난자 해동 경과일", "임신 시도 또는 마지막 임신 경과 연수"]:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isna().astype(int)

    if "시술 유형" in df.columns:
        mask = df["시술 유형"] == "DI"
        cols = [c for c in DI_ZERO_COLS if c in df.columns]
        df.loc[mask, cols] = df.loc[mask, cols].fillna(0)

    for col in ["착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부", "PGD 시술 여부", "PGS 시술 여부"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "시술 당시 나이" in df.columns:
        df["age_code"] = df["시술 당시 나이"].map(AGE_MAP)
    if "난자 기증자 나이" in df.columns:
        df["egg_donor_age_code"] = df["난자 기증자 나이"].map(DONOR_EGG_AGE_MAP)
    if "정자 기증자 나이" in df.columns:
        df["sperm_donor_age_code"] = df["정자 기증자 나이"].map(DONOR_SPERM_AGE_MAP)

    count_cols = ["총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"]
    for col in [c for c in count_cols if c in df.columns]:
        df[f"{col}_num"] = df[col].map(COUNT_MAP)

    collected = safe_num(df, "수집된 신선 난자 수")
    mixed = safe_num(df, "혼합된 난자 수")
    injected = safe_num(df, "미세주입된 난자 수")
    embryos_from_inj = safe_num(df, "미세주입에서 생성된 배아 수")
    total_embryos = safe_num(df, "총 생성 배아 수")
    transferred = safe_num(df, "이식된 배아 수")
    stored = safe_num(df, "저장된 배아 수")
    micro_trans = safe_num(df, "미세주입 배아 이식 수")
    partner_mix = safe_num(df, "파트너 정자와 혼합된 난자 수")
    donor_mix = safe_num(df, "기증자 정자와 혼합된 난자 수")
    transfer_day = safe_num(df, "배아 이식 경과일")

    df["mature_oocyte_rate_raw"] = mixed / (collected + 1)
    df["fertilization_rate_raw"] = embryos_from_inj / (injected + 1)
    df["embryo_use_rate_raw"] = transferred / (total_embryos + 1)
    df["embryo_store_rate_raw"] = stored / (total_embryos + 1)
    df["micro_transfer_rate_raw"] = micro_trans / (transferred + 1)
    df["oocyte_to_embryo_rate_raw"] = total_embryos / (collected + 1)
    df["partner_sperm_rate_raw"] = partner_mix / (mixed + 1)
    df["donor_sperm_rate_raw"] = donor_mix / (mixed + 1)
    df["has_surplus_embryo_raw"] = (stored > 0).astype(float)
    df["is_blastocyst_day5_raw"] = (transfer_day == 5).astype(float)
    df["is_long_culture_raw"] = (transfer_day >= 4).astype(float)
    df["ideal_transfer_day_raw"] = transfer_day.isin([3, 5]).astype(float)
    df["blastocyst_and_surplus_raw"] = ((transfer_day == 5) & (stored > 0)).astype(float)
    df["is_transfer_canceled_raw"] = (transferred == 0).astype(float)

    if "age_code" in df.columns and "총 시술 횟수_num" in df.columns:
        df["age_x_total_treatments_raw"] = df["age_code"] * df["총 시술 횟수_num"]
    if "총 임신 횟수_num" in df.columns and "총 출산 횟수_num" in df.columns:
        df["past_miscarriage_count_raw"] = (df["총 임신 횟수_num"] - df["총 출산 횟수_num"]).clip(lower=0)
    if "총 임신 횟수_num" in df.columns and "총 시술 횟수_num" in df.columns:
        df["past_treatment_eff_raw"] = df["총 임신 횟수_num"] / (df["총 시술 횟수_num"] + 1)
        df["repeated_failure_raw"] = ((df["총 시술 횟수_num"] >= 3) & (df["총 임신 횟수_num"] == 0)).astype(int)

    for col in df.select_dtypes(include=["object"]).columns.tolist():
        df[col] = df[col].fillna("__MISSING__").astype(str)
    return df


def load_raw_matrix():
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_fe = preprocess_raw(train_raw)
    test_fe = preprocess_raw(test_raw)
    y_raw = train_fe[TARGET].copy().astype(int)
    X_raw = train_fe.drop(columns=["ID", TARGET])
    Xt_raw = test_fe.drop(columns=["ID"]).reindex(columns=X_raw.columns)
    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    cat_features = [X_raw.columns.get_loc(c) for c in cat_cols]
    return X_raw, Xt_raw, y_raw, cat_features

BASE_CAT = {
    "iterations": 6000,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "allow_writing_files": False,
    "verbose": 0,
    "thread_count": -1,
    "early_stopping_rounds": 250,
}
CAT_PARAMS = {
    "base_depth7": {
        "iterations": 5000,
        "learning_rate": 0.025,
        "depth": 7,
        "l2_leaf_reg": 12.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.75,
        "rsm": 0.75,
        "scale_pos_weight": SPW,
        "early_stopping_rounds": 200,
    },
    "depth8_conservative": {
        "learning_rate": 0.018,
        "depth": 8,
        "l2_leaf_reg": 24.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.75,
        "rsm": 0.65,
        "scale_pos_weight": SPW,
    },
}


def cv_catboost(name, X_raw, Xt_raw, y_raw, cat_features):
    params = BASE_CAT.copy()
    params.update(CAT_PARAMS[name])
    oofs, tests = [], []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X_raw))
        test_pred = np.zeros(len(Xt_raw))
        test_pool = Pool(Xt_raw, cat_features=cat_features)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y_raw), start=1):
            train_pool = Pool(X_raw.iloc[tr_idx], y_raw.iloc[tr_idx], cat_features=cat_features)
            val_pool = Pool(X_raw.iloc[val_idx], y_raw.iloc[val_idx], cat_features=cat_features)
            model = CatBoostClassifier(**params, random_seed=seed + fold)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            pred = model.predict_proba(val_pool)[:, 1]
            oof[val_idx] = pred
            test_pred += model.predict_proba(test_pool)[:, 1] / N_FOLDS
            print(f"{name} seed={seed} fold={fold}: AUC={roc_auc_score(y_raw.iloc[val_idx], pred):.6f}, iter={model.best_iteration_}")
        print(f"{name} seed={seed} OOF={roc_auc_score(y_raw, oof):.6f}")
        oofs.append(oof)
        tests.append(test_pred)

    oof = np.mean(oofs, axis=0)
    test = np.mean(tests, axis=0)
    print(f"{name} multi-seed:", roc_auc_score(y_raw, oof))
    return oof, test


def get_catboost(name):
    loaded = maybe_load_pair(CACHE_RAW, name)
    if loaded is not None:
        return loaded
    X_raw, Xt_raw, y_raw, cat_features = load_raw_matrix()
    assert np.array_equal(y_raw.values, y.values)
    oof, test = cv_catboost(name, X_raw, Xt_raw, y_raw, cat_features)
    save_pair(CACHE_RAW, name, oof, test)
    return oof, test

compact_oof, compact_test = get_compact()
lgbm72_oof, lgbm72_test = get_lgbm72()
raw_d8_oof, raw_d8_test = get_catboost("depth8_conservative")
raw_d7_oof, raw_d7_test = get_catboost("base_depth7")
fc80_oof, fc80_test = get_feature_count_nf80()

source_board = pd.DataFrame({
    "source": ["compact", "lgbm72", "raw_d8", "raw_d7", "fc80"],
    "oof_auc": [
        roc_auc_score(y, compact_oof),
        roc_auc_score(y, lgbm72_oof),
        roc_auc_score(y, raw_d8_oof),
        roc_auc_score(y, raw_d7_oof),
        roc_auc_score(y, fc80_oof),
    ],
})
print(source_board.sort_values("oof_auc", ascending=False).to_string(index=False))

final_oof = 0.25 * compact_oof + 0.35 * lgbm72_oof + 0.30 * raw_d8_oof + 0.10 * raw_d7_oof
final_test = 0.25 * compact_test + 0.35 * lgbm72_test + 0.30 * raw_d8_test + 0.10 * raw_d7_test

review_oof = 0.30 * compact_oof + 0.32 * lgbm72_oof + 0.38 * raw_d8_oof
review_test = 0.30 * compact_test + 0.32 * lgbm72_test + 0.38 * raw_d8_test

rank_oof = 0.20 * rank01(compact_oof) + 0.28 * rank01(lgbm72_oof) + 0.38 * rank01(raw_d8_oof) + 0.14 * rank01(raw_d7_oof)
rank_test = 0.20 * rank01(compact_test) + 0.28 * rank01(lgbm72_test) + 0.38 * rank01(raw_d8_test) + 0.14 * rank01(raw_d7_test)

meta_oof = 0.85 * final_oof + 0.05 * review_oof + 0.10 * rank_oof
meta_test = 0.85 * final_test + 0.05 * review_test + 0.10 * rank_test

seg_oof = meta_oof.copy()
seg_test = meta_test.copy()

rules = [
    ("배아 이식 경과일", 1.0, lgbm72_oof, lgbm72_test, 1.00),
    ("배아 이식 경과일", 5.0, final_oof, final_test, 0.40),
    ("is_eset", 1.0, compact_oof, compact_test, 0.45),
    ("이식된 배아 수", 3.0, raw_d7_oof, raw_d7_test, 1.00),
]

for feature, value, source_oof, source_test, weight in rules:
    train_mask = (X_all[feature] == value).to_numpy()
    test_mask = (Xt_all[feature] == value).to_numpy()
    seg_oof[train_mask] = (1 - weight) * seg_oof[train_mask] + weight * source_oof[train_mask]
    seg_test[test_mask] = (1 - weight) * seg_test[test_mask] + weight * source_test[test_mask]

feature_sweep_oof = 0.40 * fc80_oof + 0.25 * compact_oof + 0.35 * raw_d8_oof
feature_sweep_test = 0.40 * fc80_test + 0.25 * compact_test + 0.35 * raw_d8_test

final_blend_oof = 0.76 * seg_oof + 0.14 * fc80_oof + 0.10 * feature_sweep_oof
final_blend_test = 0.76 * seg_test + 0.14 * fc80_test + 0.10 * feature_sweep_test

print("final_oof:", roc_auc_score(y, final_oof))
print("review_oof:", roc_auc_score(y, review_oof))
print("meta_oof:", roc_auc_score(y, meta_oof))
print("seg_oof:", roc_auc_score(y, seg_oof))
print("feature_sweep_oof:", roc_auc_score(y, feature_sweep_oof))
print("final_blend_oof:", roc_auc_score(y, final_blend_oof))

out_path = SUB_DIR / "final_submission.csv"
sub = pd.DataFrame({"ID": sample_sub["ID"], "probability": np.clip(final_blend_test, 0, 1)})

assert len(sub) == len(sample_sub)
assert sub["probability"].notna().all()
assert sub["probability"].between(0, 1).all()

sub.to_csv(out_path, index=False)
print("submission:", out_path)
sub.head()
