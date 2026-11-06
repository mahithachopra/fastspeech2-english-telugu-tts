"""Assemble the submission repo from submission_bundle.zip (downloaded from Kaggle).

Run from the repo root (the folder that contains README.md and subsample.py):
    python assemble_repo.py                      # looks for ~/Downloads/submission_bundle.zip
    python assemble_repo.py "D:\\path\\submission_bundle2.zip"
It copies English/, Telugu/, Results/, tools/ into the repo, removes reference-repo clutter,
lets git keep the generated .wav files, stages everything and prints a check. It never commits.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

repo = Path(__file__).resolve().parent
zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "submission_bundle2.zip"
if not zip_path.exists():
    sys.exit(f"Bundle not found: {zip_path}\nGive its path:  python assemble_repo.py <path-to-zip>")

tmp = repo / "bundle_tmp"
shutil.rmtree(tmp, ignore_errors=True)
print(f"extracting {zip_path.name} ({zip_path.stat().st_size / 1e6:.0f} MB) ...")
with zipfile.ZipFile(zip_path) as z:
    z.extractall(tmp)
rel = tmp / "release"
if not rel.exists():
    sys.exit("The zip does not contain a 'release' folder - is this the right file?")

for name in ("English", "Telugu", "Results", "tools"):
    shutil.copytree(rel / name, repo / name, dirs_exist_ok=True)
shutil.rmtree(tmp, ignore_errors=True)

for lang in ("English", "Telugu"):                       # reference-repo clutter
    for junk in (".gitignore", "index.html", "README.md"):
        (repo / lang / junk).unlink(missing_ok=True)
    shutil.rmtree(repo / lang / "config", ignore_errors=True)

dl = Path.home() / "Downloads"                            # README / requirements / helper downloaded from the chat
for name, dest in (("README.md", "README.md"), ("requirements.txt", "requirements.txt"),
                   ("prepare_telugu_mfa.py", "tools/prepare_telugu_mfa.py")):
    d, s = repo / dest, dl / name
    if s.exists() and (not d.exists() or d.stat().st_size < 200):
        shutil.copy(s, d)
        print(f"copied {s} -> {dest}")

helper = repo / "prepare_telugu_mfa.py"                  # belongs in tools/
if helper.exists():
    shutil.move(str(helper), str(repo / "tools" / "prepare_telugu_mfa.py"))

gi = repo / ".gitignore"                                  # keep the generated audio
lines = gi.read_text().splitlines() if gi.exists() else []
for want in ("!Results/**/*.wav", "bundle_tmp/"):
    if want not in [l.strip() for l in lines]:
        lines.append(want)
gi.write_text("\n".join(lines) + "\n")


def git(*args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout


git("add", "-A")
files = git("ls-files").splitlines()
wavs = [f for f in files if f.startswith("Results/") and f.endswith(".wav")]
ckpts = [f for f in files if "/checkpoints/" in f]
big = [(f, (repo / f).stat().st_size / 1e6) for f in files if (repo / f).exists() and (repo / f).stat().st_size > 95e6]

print("\n=== check ===")
print(f"tracked files: {len(files)}")
print(f"generated .wav tracked: {len(wavs)}   (expected 20)")
print(f"checkpoints tracked: {len(ckpts)}   (expected 4)")
for c in ckpts:
    print(f"   {c}  {(repo / c).stat().st_size / 1e6:.0f} MB")
print("files over GitHub's 100 MB limit:", big or "none")
for lang in ("English", "Telugu"):
    missing = [f for f in ("train.py", "preprocess.py", "inference.py", "config.yaml") if not (repo / lang / f).exists()]
    print(f"{lang}: missing {missing or 'nothing'}")
for f in ("README.md", "requirements.txt"):
    p = repo / f
    print(f"{f}: {p.stat().st_size if p.exists() else 0} bytes" + ("   <-- EMPTY, copy the new one here" if not p.exists() or p.stat().st_size < 200 else ""))
for d in sorted((repo / "Results").iterdir()):
    print(f"Results/{d.name}: {len(list(d.rglob('*.wav')))} wav, {len(list(d.rglob('*.png')))} png")
print("\nNothing has been committed yet.")
