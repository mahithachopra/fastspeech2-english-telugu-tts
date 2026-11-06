import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [("Loss/total_loss", "total"), ("Loss/mel_loss", "mel"), ("Loss/mel_postnet_loss", "mel (postnet)"),
        ("Loss/pitch_loss", "pitch"), ("Loss/energy_loss", "energy"), ("Loss/duration_loss", "duration")]


def load(folder):
    ea = EventAccumulator(folder, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for t, _ in TAGS:
        if t in ea.Tags()["scalars"]:
            by_step = {e.step: e.value for e in ea.Scalars(t)}   # a resumed run rewrites some steps: keep the latest value
            steps = sorted(by_step)
            out[t] = (steps, [by_step[s] for s in steps])
    return out


def smooth(v, k=5):
    out = []
    for i in range(len(v)):
        w = v[max(0, i - k + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def draw(data, title, path, do_smooth):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (tag, label) in zip(axes.ravel(), TAGS):
        if tag in data:
            s, v = data[tag]
            if do_smooth:
                ax.plot(s, v, color="#bbbbbb", lw=0.8)
            ax.plot(s, smooth(v) if do_smooth else v, color="#4c78a8", lw=1.5)
        ax.set_title(label + " loss")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="output/log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr, va = load(os.path.join(a.log, "train")), load(os.path.join(a.log, "val"))
    draw(tr, f"{a.title} training loss", os.path.join(a.out, "training_loss.png"), True)
    draw(va, f"{a.title} validation loss", os.path.join(a.out, "validation_loss.png"), False)
    for name, d in (("train", tr), ("val", va)):
        s, v = d["Loss/total_loss"]
        print(f"{name}: {len(s)} points, first {v[0]:.3f} -> last {v[-1]:.3f}, min {min(v):.3f} at step {s[v.index(min(v))]}")
    print("saved to", a.out)
