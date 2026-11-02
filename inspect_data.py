from datasets import load_dataset, Audio

for name, split in [("ai4bharat/Svarah", "test"),
                    ("SPRINGLab/IndicTTS_Telugu", "train")]:
    ds = load_dataset(name, split=split)
    audio_cols = [c for c, f in ds.features.items() if isinstance(f, Audio)]
    print("\n==", name, "==")
    print(ds)
    print("audio columns:", audio_cols)
    raw = ds.cast_column(audio_cols[0], Audio(decode=False))
    for i in (0, 1, 2):
        row = raw[i]
        print({k: v for k, v in row.items() if k != audio_cols[0]})
        print("audio path:", row[audio_cols[0]]["path"])
    if "gender" in ds.column_names:
        print(ds.to_pandas()["gender"].value_counts())