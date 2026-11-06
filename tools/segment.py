"""Cut long utterances at pauses into 2-7 s segments, reusing the existing MFA TextGrids (no re-alignment).

    python segment.py <raw_dir> <aligned_dir> <manifest_in.csv> <out_root> <Language>
"""
import csv
import glob
import os
import sys

import soundfile as sf
import tgt

MIN_LEN, MAX_LEN, PAUSE, DROP_BELOW, DROP_ABOVE = 2.0, 7.0, 0.08, 0.6, 11.0
SIL = {"", "sp", "spn", "sil"}


def cut_points(cands, total):
    cuts, start, best = [], 0.0, None
    for c in cands:
        broke = False
        while c - start > MAX_LEN:
            if best is None:              # no earlier pause available: cut here
                cuts.append(c)
                start, broke = c, True
                break
            cuts.append(best)
            start, best = best, None
        if broke:
            continue
        if c - start >= MIN_LEN:
            best = c
    bounds = [0.0] + cuts + [total]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < MIN_LEN:   # merge a too-short tail
        del bounds[-2]
    return bounds


def main(raw, aligned, manifest_in, out_root, lang):
    split = {r["id"]: r["split"] for r in csv.DictReader(open(manifest_in, encoding="utf-8"))}
    rows, hours = [], {"train": 0.0, "val": 0.0}
    n_in = 0
    for tg_path in sorted(glob.glob(f"{aligned}/*/*.TextGrid")):
        spk = os.path.basename(os.path.dirname(tg_path))
        base = os.path.basename(tg_path)[: -len(".TextGrid")]
        wav_path = f"{raw}/{spk}/{base}.wav"
        if base not in split or not os.path.exists(wav_path):
            continue
        n_in += 1
        tg = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        ph, wd = tg.get_tier_by_name("phones"), tg.get_tier_by_name("words")
        y, sr = sf.read(wav_path)
        total = len(y) / sr
        cands = [(i.start_time + i.end_time) / 2 for i in ph.intervals
                 if i.text in SIL and i.end_time - i.start_time >= PAUSE and 0 < i.start_time and i.end_time < total - 0.05]
        bounds = cut_points(cands, total)
        for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            if not DROP_BELOW <= b - a <= DROP_ABOVE:
                continue
            sid = f"{base}_s{k:02d}"
            words = [w.text for w in wd.intervals if w.text and a <= (w.start_time + w.end_time) / 2 < b]
            if not words:
                continue
            for d in (f"{out_root}/raw/{spk}", f"{out_root}/aligned/{spk}"):
                os.makedirs(d, exist_ok=True)
            sf.write(f"{out_root}/raw/{spk}/{sid}.wav", y[int(a * sr): int(b * sr)], sr)
            open(f"{out_root}/raw/{spk}/{sid}.lab", "w", encoding="utf-8").write(" ".join(words))
            new = tgt.TextGrid()
            for tier in (ph, wd):
                t = tgt.IntervalTier(start_time=0.0, end_time=b - a, name=tier.name)
                for i in tier.intervals:
                    s, e = max(i.start_time, a) - a, min(i.end_time, b) - a
                    if e - s > 1e-4:
                        t.add_interval(tgt.Interval(s, e, i.text))
                new.add_tier(t)
            tgt.io.write_to_file(new, f"{out_root}/aligned/{spk}/{sid}.TextGrid", format="long")
            rows.append({"id": sid, "speaker": spk, "split": split[base], "duration": round(b - a, 3)})
            hours[split[base]] += (b - a) / 3600
    with open(f"{out_root}/{lang}_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "speaker", "split", "duration"])
        w.writeheader()
        w.writerows(rows)
    d = sorted(r["duration"] for r in rows)
    print(f"{lang}: {n_in} utterances -> {len(rows)} segments | train {hours['train']:.2f} h, val {hours['val']:.2f} h "
          f"| segment length median {d[len(d) // 2]:.1f}s, max {d[-1]:.1f}s")


if __name__ == "__main__":
    main(*sys.argv[1:6])
# END OF FILE
