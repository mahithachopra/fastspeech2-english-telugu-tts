"""Clean the Telugu .lab transcripts and build the character-level pronunciation dictionary for MFA.

    python tools/prepare_telugu_mfa.py raw_data/Telugu telugu.dict

Every word is spelled as its sequence of Unicode code points (Telugu is close to phonetic, so
each code point -- consonant, vowel sign, virama, anusvara -- is one "phone"). MFA then learns
acoustic models for these symbols from the 6 h subset, with no pretrained Telugu aligner needed.
"""
import glob
import re
import sys
import unicodedata

corpus, out_dict = sys.argv[1], sys.argv[2]
words = set()
labs = glob.glob(f"{corpus}/**/*.lab", recursive=True)
for p in labs:
    t = unicodedata.normalize("NFC", open(p, encoding="utf-8").read())
    t = re.sub(r"[^ఀ-౿\s]", " ", t)       # keep Telugu letters only (drops punctuation, digits, Latin)
    t = re.sub(r"[​‌‍]", "", t)      # zero-width characters
    t = " ".join(t.split())
    open(p, "w", encoding="utf-8").write(t)
    words.update(t.split())
with open(out_dict, "w", encoding="utf-8") as f:
    for w in sorted(words):
        f.write(w + "\t" + " ".join(list(w)) + "\n")
print(len(labs), "transcripts,", len(words), "distinct words ->", out_dict)
