import re
import unicodedata


def remove_accents(text):
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKD", text)

    return "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )


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
    "corp": "corporation",
    "ltd": "limited",
    "pvt": "private",
}

ADDRESS_ABBREVIATIONS = {
    "corp": "corporation",
    "ltd": "limited",
    "pvt": "private",
    "rd": "road",
    "st": "street",
    "rue": "rue",
    "av": "avenue",
    "bd": "boulevard",
}


def expand_abbreviations(text, abbreviations):
    tokens = text.split()

    expanded = [
        abbreviations.get(token, token)
        for token in tokens
    ]

    return " ".join(expanded)


def normalize_name(s):
    text = normalize_text(s)

    return expand_abbreviations(
        text,
        NAME_ABBREVIATIONS
    )


def normalize_address(s):
    text = normalize_text(s)

    return expand_abbreviations(
        text,
        ADDRESS_ABBREVIATIONS
    )


def extract_postal(s):
    text = normalize_text(s)

    matches = re.findall(r"\b\d{5,6}\b", text)

    if not matches:
        return ""

    return matches[-1]


def tokenize(s):
    text = normalize_text(s)

    if not text:
        return set()

    return set(text.split())


if __name__ == "__main__":
    print("Name normalization tests:")

    name_tests = [
        "Café de la Gare",
        "ACME Corp.",
        "Société Générale",
    ]

    for text in name_tests:
        print(f"{text!r} -> {normalize_name(text)!r}")

    print("\nAddress normalization tests:")

    address_tests = [
        "12 MG Rd, Bengaluru",
        "25 St John's Road, Delhi",
        "10 Av de Paris",
        "5 Bd Haussmann",
    ]

    for text in address_tests:
        print(f"{text!r} -> {normalize_address(text)!r}")

    print("\nPostal tests:")

    print(
        "Bengaluru 560001 ->",
        extract_postal("Bengaluru 560001")
    )

    print(
        "75001 Paris ->",
        extract_postal("75001 Paris")
    )