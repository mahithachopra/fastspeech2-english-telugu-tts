import io
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

# ---------- Svarah ----------
ds = load_dataset("ai4bharat/Svarah", split="test")
meta = ds.select_columns(["duration", "gender"]).to_pandas()
raw = ds.cast_column("audio_filepath", Audio(decode=False))
meta["path"] = [a["path"] for a in raw["audio_filepath"]]
meta["speaker"] = meta["path"].str.extract(r"_([fm]\d+)_chunk")[0]

print("== Svarah ==")
print("rows:", len(meta), "| unparsed speaker:", meta["speaker"].isna().sum())
print("speakers:", meta["speaker"].nunique())
print("total hours:", round(meta["duration"].sum() / 3600, 2))
print(meta["duration"].describe().round(2))

for lo, hi in [(1, 15), (1.5, 12), (2, 12)]:
    k = meta[(meta.duration >= lo) & (meta.duration <= hi)]
    print(f"kept {lo}-{hi}s: {len(k)} rows, {k.duration.sum()/3600:.2f} h, "
          f"{k.speaker.nunique()} speakers")

k = meta[(meta.duration >= 1) & (meta.duration <= 15)]
sp = k.groupby("speaker")["duration"].sum().div(60).round(1).sort_values()
print("minutes per speaker (1-15s):")
print(sp.describe().round(1))

# ---------- Telugu ----------
tds = load_dataset("SPRINGLab/IndicTTS_Telugu", split="train")
tmeta = tds.select_columns(["gender"]).to_pandas()
traw = tds.cast_column("audio", Audio(decode=False))["audio"]
tmeta["path"] = [a["path"] for a in traw]
tmeta["dur"] = [sf.info(io.BytesIO(a["bytes"])).duration for a in traw]
tmeta["sr"] = [sf.info(io.BytesIO(a["bytes"])).samplerate for a in traw]

print("\n== Telugu ==")
print(pd.crosstab(tmeta["gender"], tmeta["path"].str.contains("female")))
f = tmeta[tmeta["gender"] == 0]
print("female rows:", len(f), "| hours:", round(f.dur.sum() / 3600, 2))
print("sample rates:", f.sr.unique())
print(f.dur.describe().round(2))
for lo, hi in [(1, 12), (1.5, 15)]:
    k = f[(f.dur >= lo) & (f.dur <= hi)]
    print(f"kept {lo}-{hi}s: {len(k)} rows, {k.dur.sum()/3600:.2f} h")