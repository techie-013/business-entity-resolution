import unicodedata
import re
import pandas as pd

# ─────────────────────────────────────────────
# Abbreviation map — US / India / France
# ─────────────────────────────────────────────
ABBREV = {
    # English / US legal
    "corp": "corporation", "co": "company", "ltd": "limited",
    "inc": "incorporated", "llc": "limited liability company",
    "pvt": "private", "ste": "suite",
    # Address (US + India)
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "sq": "square", "ln": "lane", "dr": "drive", "hwy": "highway",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "no": "number", "&": "and",
    # France
    "r": "rue", "av": "avenue", "bd": "boulevard", "pl": "place",
    "che": "chemin", "imp": "impasse", "all": "allee",
    "sarl": "sarl", "sas": "sas", "sa": "sa", "eurl": "eurl", "snc": "snc",
    "ste": "societe", "cie": "compagnie",
}


def normalize_text(s):
    """Lowercase + strip accents + remove punctuation + collapse spaces."""
    if pd.isna(s) or s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def expand_abbrev(text):
    """Replace abbreviations. Safe for NaN / non-strings."""
    if not isinstance(text, str):
        return ""
    return " ".join(ABBREV.get(t, t) for t in text.split())


def normalize_name(name):
    return expand_abbrev(normalize_text(name))


def normalize_address(addr):
    return expand_abbrev(normalize_text(addr))


def extract_postal(text):
    """Extract 5–6 digit postal code (US ZIP, India PIN, France)."""
    if pd.isna(text) or not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    m = re.search(r"\b(\d{5,6})(?:-\d{4})?\b", text)
    return m.group(1) if m else ""


def tokenize(s):
    return set(normalize_text(s).split()) if s else set()