import re
import unicodedata
from functools import lru_cache


def remove_accents(text):
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_text(s):
    if s is None:
        return ""
    if isinstance(s, float) and s != s:
        return ""
    text = str(s)
    text = unicodedata.normalize("NFKC", text)
    text = remove_accents(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


NAME_ABBREVIATIONS = {
    "corp": "corporation", "co": "company", "ltd": "limited",
    "inc": "incorporated", "llc": "limited liability company",
    "pvt": "private", "pte": "private",
    "sarl": "sarl", "sas": "sas", "sa": "sa", "eurl": "eurl", "snc": "snc",
    "ste": "societe", "cie": "compagnie",
    "intl": "international", "svc": "service", "svcs": "services",
    "mfg": "manufacturing", "tech": "technology",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "sq": "square", "ln": "lane",
    "dr": "drive", "hwy": "highway",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "no": "number", "&": "and",
    "r": "rue", "rue": "rue", "bd": "boulevard", "pl": "place",
    "che": "chemin", "imp": "impasse", "all": "allee",
}


def expand_abbreviations(text, abbreviations):
    if not isinstance(text, str):
        return ""
    return " ".join(abbreviations.get(tok, tok) for tok in text.split())


@lru_cache(maxsize=2_000_000)
def _cached_norm_name(s):
    return expand_abbreviations(normalize_text(s), NAME_ABBREVIATIONS)


@lru_cache(maxsize=2_000_000)
def _cached_norm_addr(s):
    return expand_abbreviations(normalize_text(s), ADDRESS_ABBREVIATIONS)


@lru_cache(maxsize=2_000_000)
def _cached_postal(s):
    m = re.search(r"\b(\d{5})(?:-?\d{4})?\b", str(s))
    return m.group(1) if m else ""


def normalize_name(s):
    if s is None or (isinstance(s, float) and s != s):
        return ""
    return _cached_norm_name(str(s))


def normalize_address(s):
    if s is None or (isinstance(s, float) and s != s):
        return ""
    return _cached_norm_addr(str(s))


def extract_postal(s):
    if s is None or (isinstance(s, float) and s != s):
        return ""
    return _cached_postal(str(s))


def tokenize(s):
    text = normalize_text(s)
    return set(text.split()) if text else set()