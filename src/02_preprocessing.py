from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


TARGET = "임신 성공 여부"
COUNT_MAP = {"0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6}
AGE_MAP = {
    "만18-34세": 0,
    "만35-37세": 1,
    "만38-39세": 2,
    "만40-42세": 3,
    "만43-44세": 4,
    "만45-50세": 5,
    "알 수 없음": 2,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["포배기_이식"] = (df["배아 이식 경과일"] == 5).astype(float)
    df["장기배양_이식"] = (df["배아 이식 경과일"] >= 4).astype(float)
    df["저반응_여부"] = (df["수집된 신선 난자 수"] <= 3).astype(float)
    df["고반응_여부"] = (df["수집된 신선 난자 수"] > 15).astype(float)
    df["성숙_난자율"] = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)
    df["수정률"] = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + 1)
    df["배아_활용률"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)
    df["잉여배아_존재"] = (df["저장된 배아 수"] > 0).astype(float)
    df["미세주입_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)

    treatments = df["총 시술 횟수"].map(COUNT_MAP)
    pregnancies = df["총 임신 횟수"].map(COUNT_MAP)
    births = df["총 출산 횟수"].map(COUNT_MAP)
    ivf_births = df["IVF 출산 횟수"].map(COUNT_MAP)

    df["과거_유산횟수"] = (pregnancies - births).clip(lower=0)
    df["반복실패_여부"] = ((treatments >= 3) & (pregnancies == 0)).astype(int)
    df["IVF_성공이력"] = (ivf_births > 0).astype(int)
    df["과거_시술효율"] = pregnancies / (treatments + 1)

    df["기증_난자"] = (df["난자 출처"] == "기증 제공").astype(int)
    df["기증_정자"] = (df["정자 출처"] == "기증 제공").astype(int)
    df["배란유도_기록됨"] = (~df["배란 유도 유형"].isin(["기록되지 않은 시행", "알 수 없음"])).astype(int)
    df["현재시술_목적"] = (df["배아 생성 주요 이유"] == "현재 시술용").astype(int)
    df["포배기_AND_잉여배아"] = ((df["배아 이식 경과일"] == 5) & (df["저장된 배아 수"] > 0)).astype(float)
    df["난자_배아_전환율"] = df["총 생성 배아 수"] / (df["수집된 신선 난자 수"] + 1)
    df["배아_저장비율"] = df["저장된 배아 수"] / (df["총 생성 배아 수"] + 1)
    df["미세주입_저장비율"] = df["미세주입 후 저장된 배아 수"] / (df["미세주입에서 생성된 배아 수"] + 1)
    df["파트너정자_비율"] = df["파트너 정자와 혼합된 난자 수"] / (df["혼합된 난자 수"] + 1)
    df["신선_배아_배양시간"] = (df["배아 이식 경과일"] - df["난자 혼합 경과일"]).fillna(0)
    df["이상적_배양기간"] = df["배아 이식 경과일"].isin([3.0, 5.0]).astype(int)

    male_cols = [
        "불임 원인 - 남성 요인",
        "불임 원인 - 정자 농도",
        "불임 원인 - 정자 면역학적 요인",
        "불임 원인 - 정자 운동성",
        "불임 원인 - 정자 형태",
    ]
    female_cols = [
        "불임 원인 - 난관 질환",
        "불임 원인 - 배란 장애",
        "불임 원인 - 자궁경부 문제",
        "불임 원인 - 자궁내막증",
    ]
    df["남성_불임_심각도"] = df[male_cols].sum(axis=1)
    df["여성_불임_심각도"] = df[female_cols].sum(axis=1)

    age_code = df["시술 당시 나이"].map(AGE_MAP).fillna(2)
    donor_egg = df["난자 출처"] == "기증 제공"
    donor_map = {"알 수 없음": -1, "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2, "만31-35세": 3}
    donor_age_code = df["난자 기증자 나이"].map(donor_map).fillna(-1)
    treatment_code = df["총 시술 횟수"].map(COUNT_MAP).fillna(0)

    df["실효_난자나이"] = np.where(donor_egg & (donor_age_code != -1), donor_age_code, age_code)
    df["고령_기증난자_상쇄"] = ((age_code >= 3) & donor_egg).astype(int)
    df["혼합난자_배아효율"] = df["총 생성 배아 수"] / (df["혼합된 난자 수"] + 1)
    df["초고령_43이상"] = (age_code >= 4).astype(int)
    df["연령_x_시술횟수"] = age_code * treatment_code
    df["is_blastocyst"] = df["특정 시술 유형"].astype(str).str.upper().str.contains("BLASTOCYST", na=False).astype(int)
    df["is_transfer_canceled"] = (df["이식된 배아 수"].fillna(0) == 0).astype(int)
    df["is_eset"] = (df["단일 배아 이식 여부"].fillna(0) == 1).astype(int)
    cause_cols = [c for c in df.columns if "불임 원인 - " in c]
    df["cause_count"] = df[cause_cols].fillna(0).astype(int).sum(axis=1)
    df["is_mix_date_missing"] = ((df["시술 유형"].astype(str) == "IVF") & df["난자 혼합 경과일"].isnull()).astype(int)

    df["나이_x_시술"] = age_code * treatment_code
    df["배아생성_능력"] = df["수집된 신선 난자 수"] * df["난자_배아_전환율"]
    return df


def preprocess(data_dir: Path) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    drop_cols = ["불임 원인 - 여성 요인", "난자 채취 경과일"]
    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)

    di_zero_cols = [
        "미세주입된 난자 수",
        "미세주입에서 생성된 배아 수",
        "총 생성 배아 수",
        "이식된 배아 수",
        "미세주입 배아 이식 수",
        "저장된 배아 수",
        "미세주입 후 저장된 배아 수",
        "해동된 배아 수",
        "해동 난자 수",
        "수집된 신선 난자 수",
        "저장된 신선 난자 수",
        "혼합된 난자 수",
        "파트너 정자와 혼합된 난자 수",
        "기증자 정자와 혼합된 난자 수",
        "난자 혼합 경과일",
        "배아 이식 경과일",
        "배아 해동 경과일",
        "동결 배아 사용 여부",
        "신선 배아 사용 여부",
        "기증 배아 사용 여부",
        "단일 배아 이식 여부",
        "임신 시도 또는 마지막 임신 경과 연수",
    ]
    di_zero_cols = [c for c in di_zero_cols if c in train.columns]
    train.loc[train["시술 유형"] == "DI", di_zero_cols] = train.loc[train["시술 유형"] == "DI", di_zero_cols].fillna(0)
    test.loc[test["시술 유형"] == "DI", di_zero_cols] = test.loc[test["시술 유형"] == "DI", di_zero_cols].fillna(0)

    for col in ["착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부"]:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    for col in ["난자 해동 경과일", "배아 해동 경과일", "난자 혼합 경과일", "배아 이식 경과일", "임신 시도 또는 마지막 임신 경과 연수"]:
        train[f"{col}_결측"] = train[col].isnull().astype(int)
        test[f"{col}_결측"] = test[col].isnull().astype(int)

    for df in (train, test):
        df.loc[df["이식된 배아 수"].fillna(0).eq(0), "배아 이식 경과일"] = df.loc[
            df["이식된 배아 수"].fillna(0).eq(0), "배아 이식 경과일"
        ].fillna(0)
        df.loc[df["혼합된 난자 수"].fillna(0).eq(0), "난자 혼합 경과일"] = df.loc[
            df["혼합된 난자 수"].fillna(0).eq(0), "난자 혼합 경과일"
        ].fillna(0)
        df.loc[df["해동된 배아 수"].fillna(0).eq(0), "배아 해동 경과일"] = df.loc[
            df["해동된 배아 수"].fillna(0).eq(0), "배아 해동 경과일"
        ].fillna(0)

    num_cols = train.drop(columns=["ID", TARGET], errors="ignore").select_dtypes(include=np.number).columns
    median_val = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(median_val)
    test[num_cols] = test[num_cols].fillna(median_val)

    cat_cols = train.drop(columns=["ID", TARGET], errors="ignore").select_dtypes(include="object").columns
    mode_val = train[cat_cols].mode().iloc[0]
    train[cat_cols] = train[cat_cols].fillna(mode_val)
    test[cat_cols] = test[cat_cols].fillna(mode_val)

    train = add_features(train)
    test = add_features(test)

    ordinal_mappings = {
        "시술 당시 나이": AGE_MAP,
        "총 시술 횟수": COUNT_MAP,
        "클리닉 내 총 시술 횟수": COUNT_MAP,
        "IVF 시술 횟수": COUNT_MAP,
        "DI 시술 횟수": COUNT_MAP,
        "총 임신 횟수": COUNT_MAP,
        "IVF 임신 횟수": COUNT_MAP,
        "DI 임신 횟수": COUNT_MAP,
        "총 출산 횟수": COUNT_MAP,
        "IVF 출산 횟수": COUNT_MAP,
        "DI 출산 횟수": COUNT_MAP,
        "난자 기증자 나이": {"알 수 없음": -1, "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2, "만31-35세": 3},
        "정자 기증자 나이": {"알 수 없음": -1, "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2, "만31-35세": 3, "만36-40세": 4, "만41-45세": 5},
    }
    for col, mapping in ordinal_mappings.items():
        if col in train.columns:
            train[col] = train[col].map(mapping)
            test[col] = test[col].map(mapping)

    label_cols = ["시술 시기 코드", "시술 유형", "특정 시술 유형", "배란 유도 유형", "배아 생성 주요 이유", "난자 출처", "정자 출처"]
    label_cols = [c for c in label_cols if c in train.columns]
    for col in label_cols:
        encoder = LabelEncoder()
        encoder.fit(train[col].astype(str))
        mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
        train[col] = encoder.transform(train[col].astype(str))
        test[col] = test[col].astype(str).map(mapping).fillna(-1).astype(int)

    x_train = train.drop(columns=["ID", TARGET])
    y_train = train[TARGET]
    x_test = test.drop(columns=["ID"])
    test_ids = test["ID"]

    assert x_train.isnull().sum().sum() == 0
    assert x_test.isnull().sum().sum() == 0
    assert len(x_train.select_dtypes(exclude=np.number).columns) == 0
    assert list(x_train.columns) == list(x_test.columns)

    x_train.to_csv(data_dir / "X_train.csv", index=False)
    y_train.to_csv(data_dir / "y_train.csv", index=False)
    x_test.to_csv(data_dir / "X_test.csv", index=False)
    test_ids.to_csv(data_dir / "test_ids.csv", index=False)
    print(f"saved: {x_train.shape}, {x_test.shape}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    preprocess(repo_root / "data")
