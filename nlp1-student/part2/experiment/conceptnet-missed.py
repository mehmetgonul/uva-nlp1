"""Save a plain-text breakdown of ConceptNet Numberbatch misses on SNLI train."""

import argparse
import gzip
import re
import sys
from collections import Counter
from pathlib import Path

from spellchecker import SpellChecker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from part2.clslib import BasicTokenizer
from part2.lib.dataset import DEFAULT_CACHE_DIR, FIELDS, download, load_snli
from part2.lib.embeddings import CONCEPTNET

NUMBER = re.compile(r"""[+−~≈]?[$£€¥]?(?:\d+(?:[.,:/′'″]\d+)*|[.,]\d+)[%°′'″"”’]?""")
ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)", re.IGNORECASE)
CONTRACTIONS = {"n't", "'re", "'ve", "'ll", "'d", "'m"}
REMAINING = "Remaining forms without digits"
CATEGORIES = (
    ("Hyphenated tokens", lambda token: "-" in token),
    ("The token 's", lambda token: token == "'s"),
    ("Digits only", str.isdecimal),
    ("Common contraction fragments other than 's", CONTRACTIONS.__contains__),
    ("Formatted numbers and times", lambda token: NUMBER.fullmatch(token) or token.isnumeric()),
    ("Ordinals", ORDINAL.fullmatch),
    ("Other forms containing digits", lambda token: any(character.isdigit() for character in token)),
    (REMAINING, lambda token: True),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    path = download(CONCEPTNET.url, args.cache_dir / CONCEPTNET.filename)
    # ponytail: Numberbatch uses a header followed by one space-delimited word per row.
    # Revisit this reader if the checkpoint format changes.
    with gzip.open(path, "rb") as source:
        next(source)
        vocabulary = {line.split(b" ", 1)[0] for line in source}

    tokenizer = BasicTokenizer(lowercase=False, remove_punct=False, remove_stopwords=False)
    sentences = Counter(
        pair[field] for _, pair in load_snli(split="train", cache_dir=args.cache_dir) for field in FIELDS
    )
    groups = {name: Counter() for name, _ in CATEGORIES}
    total_tokens = 0
    for sentence, occurrences in sentences.items():
        for token in tokenizer.tokens(sentence):
            if not any(character.isalnum() for character in token):
                continue
            total_tokens += occurrences
            if token.encode() in vocabulary or token.lower().encode() in vocabulary:
                continue
            category = next(name for name, matches in CATEGORIES if matches(token))
            groups[category][token] += occurrences

    remaining = groups[REMAINING]
    dictionary = SpellChecker(language="en").word_frequency.dictionary
    flagged = Counter({
        token: count for token, count in remaining.items() if token.lower() not in dictionary
    })
    groups[REMAINING] = remaining - flagged
    groups["Potential misspellings (pyspellchecker)"] = flagged

    total_misses = sum(group.total() for group in groups.values())
    miss_rate = total_misses / total_tokens if total_tokens else 0
    category_width = max(map(len, groups))
    lines = [
        f"Out-of-vocabulary breakdown for {CONCEPTNET.name} — SNLI train set",
        "",
        f"{'Category':<{category_width}}  {'Misses (% of misses)':>20}  Examples",
    ]
    for name, group in groups.items():
        count = group.total()
        share = count / total_misses if total_misses else 0
        misses = f"{count:,} ({share:.2%})"
        examples = "" if group is flagged else ", ".join(token for token, _ in group.most_common(3))
        lines.append(f"{name:<{category_width}}  {misses:>20}  {examples}".rstrip())

    total = f"{total_misses:,} ({100 if total_misses else 0}%)"
    lines.extend([
        f"{'Total':<{category_width}}  {total:>20}",
        "",
        f"{total_misses:,} of {total_tokens:,} tokens were unmatched ({miss_rate:.2%}).",
    ])

    examples = ", ".join(token for token, _ in flagged.most_common(5)) or "none"
    lines.extend([
        "",
        "The English spellchecker flags forms outside its dictionary, including some valid forms.",
        f"Most frequent flags: {examples}",
    ])

    output = Path(__file__).resolve().parent / "results" / "conceptnet-missed.txt"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
