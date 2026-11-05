import re

from num2words import num2words


def words(n, **kw):
    return num2words(n, **kw).replace("-", " ").replace(",", "")


def clock(m):
    h, mi = int(m[1]), int(m[2])
    t = words(h) + ((" oh " if mi < 10 else " ") + words(mi) if mi else " o'clock")
    return t + " " + " ".join(m[3].upper())


def normalize_en(text):
    """Times, years and numbers -> words (hyphens removed: the frontend would turn them into pauses)."""
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp][Mm])\b", clock, text)
    text = re.sub(r"\b(1\d{3}|20\d{2})\b", lambda m: words(int(m[1]), to="year"), text)
    return re.sub(r"\d+", lambda m: words(int(m[0])), text)
