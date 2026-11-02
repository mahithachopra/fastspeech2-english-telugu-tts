import io
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

# ---------- Svarah: speaker fingerprint ----------
ds = load_dataset("ai4bharat/Svarah", split="test")
cols = ["gender", "age-group", "primary_language", "native_place_state",
        "native_place_district", "highest_qualification",
        "job_category", "occupation_domain"]
m = ds.select_columns(["duration"] + cols).to_pandas()
m["spk"] = m[cols].astype(str).agg("|".join, axis=1)

print("== Svarah ==")
print("unique fingerprints:", m["spk"].nunique())
hrs = m.groupby("spk")["duration"].sum().div(3600)
print("hours per fingerprint:")
print(hrs.describe().round(3))

k = m[(m.duration >= 1) & (m.duration <= 15)]
print("kept 1-15s:", len(k), "rows,", round(k.duration.sum() / 3600, 2), "h,",
      k["spk"].nunique(), "fingerprints")

raw = ds.cast_column("audio_filepath", Audio(decode=False))["audio_filepath"][:50]
print("sample rates:", {sf.info(io.BytesIO(a["bytes"])).samplerate for a in raw})

# ---------- Telugu: length thresholds ----------
tds = load_dataset("SPRINGLab/IndicTTS_Telugu", split="train")
tm = tds.select_columns(["gender"]).to_pandas()
traw = tds.cast_column("audio", Audio(decode=False))["audio"]
tm["dur"] = [sf.info(io.BytesIO(a["bytes"])).duration for a in traw]
f = tm[tm["gender"] == 0]

print("\n== Telugu female ==")
for hi in (15, 20, 25, 30, 40):
    k = f[(f.dur >= 1.5) & (f.dur <= hi)]
    print(f"1.5-{hi}s: {len(k)} rows, {k.dur.sum()/3600:.2f} h")