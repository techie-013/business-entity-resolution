import pandas as pd
from pathlib import Path
from src.normalize import normalize_name, normalize_address, extract_postal


def _enrich(df, label=""):
    df = df.copy()
    print(f"    [{label}] normalizing names...")
    df["norm_name"] = df["business_name"].map(normalize_name)
    print(f"    [{label}] normalizing addresses...")
    df["norm_addr"] = df["business_address"].map(normalize_address)
    print(f"    [{label}] extracting postal codes...")
    df["postal"] = df["business_address"].map(extract_postal)
    df["source"] = df["entity_id"].str[:2]
    df["country"] = df["country"].astype(str).str.strip()
    return df


def load_source(path, label=""):
    print(f"  Reading {path} ...")
    df = pd.read_csv(path, sep="\t")
    print(f"  Loaded {len(df):,} rows")
    return _enrich(df, label=label)


def load_train(data_dir):
    d = Path(data_dir)
    print("[1/4] Loading Source 1 (reference)...")
    s1 = load_source(d / "train_source1.tsv", label="S1")
    print("[2/4] Loading Source 2...")
    s2 = load_source(d / "train_source2.tsv", label="S2")
    print("[3/4] Loading Source 3...")
    s3 = load_source(d / "train_source3.tsv", label="S3")
    print("[4/4] Loading ground truth...")
    gt = pd.read_csv(d / "train_ground_truth.tsv", sep="\t")
    print(f"  GT rows: {len(gt):,}")
    return s1, s2, s3, gt


def load_test(data_dir):
    d = Path(data_dir)
    print("[1/3] Loading test Source 1...")
    s1 = load_source(d / "test_source1.tsv", label="S1")
    print("[2/3] Loading test Source 2...")
    s2 = load_source(d / "test_source2.tsv", label="S2")
    print("[3/3] Loading test Source 3...")
    s3 = load_source(d / "test_source3.tsv", label="S3")
    return s1, s2, s3


def parse_ground_truth(gt):
    out = {}
    for _, r in gt.iterrows():
        raw = r["matched_entity_ids"]
        out[r["source1_entity_id"]] = (
            set() if pd.isna(raw) or raw == "" else set(raw.split(","))
        )
    return out