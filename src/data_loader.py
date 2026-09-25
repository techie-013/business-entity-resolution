import pandas as pd
import time
from pathlib import Path
from src.normalize import normalize_name, normalize_address, extract_postal


def _enrich(df, label=""):
    print(f"    [{label}] enriching {len(df):,} rows...")
    t = time.time()
    df = df.copy()
    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_addr"] = df["business_address"].map(normalize_address)
    df["postal"] = df["business_address"].map(extract_postal)
    df["country_l"] = df["country"].astype(str).str.strip().str.lower()
    print(f"    [{label}] enriched in {time.time()-t:.1f}s")
    return df


def load_source(path, label=""):
    path = Path(path)
    cache_path = Path(str(path) + ".pkl")

    if cache_path.exists():
        print(f"  Loading cached {cache_path.name} ...")
        t = time.time()
        df = pd.read_pickle(cache_path)
        print(f"  Loaded {len(df):,} rows from cache in {time.time()-t:.1f}s")
        return df

    print(f"  Reading {path} ...")
    df = pd.read_csv(path, sep="\t")
    print(f"  Loaded {len(df):,} rows")

    df = _enrich(df, label=label)

    print(f"  Saving cache → {cache_path.name} ...")
    df.to_pickle(cache_path)
    return df


def load_train(data_dir):
    d = Path(data_dir)
    print("[1/4] Loading Source 1...")
    s1 = load_source(d / "train_source1.tsv", label="S1")
    print("[2/4] Loading Source 2...")
    s2 = load_source(d / "train_source2.tsv", label="S2")
    print("[3/4] Loading Source 3...")
    s3 = load_source(d / "train_source3.tsv", label="S3")
    print("[4/4] Loading Ground Truth...")
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