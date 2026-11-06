"""Build English/ and Telugu/ FastSpeech2 project folders from a clean clone.
Usage (Kaggle):  python setup_fs2.py /kaggle/working/FastSpeech2 /kaggle/working/project
Run after MFA alignment. Expects /kaggle/working/raw/{English,Telugu} and aligned/{English,Telugu}.
"""
import os, re, shutil, sys, textwrap

SRC, DST = sys.argv[1], sys.argv[2]
RAW = "/kaggle/working/raw"
ALIGNED = "/kaggle/working/aligned"
MANIFESTS = {l: f"/kaggle/working/seg/{l}/{l}_manifest.csv" for l in ("English", "Telugu")}

# ---------------------------------------------------------------- patches
def patch_preprocessor(path):
    s = open(path, encoding="utf-8").read()
    # 1. MFA 3.x writes silence as "" -> map to "sp", treat as silence
    s = s.replace('sil_phones = ["sil", "sp", "spn"]', 'sil_phones = ["sil", "sp", "spn", ""]')
    s = s.replace("                phones.append(p)\n\n            durations.append(",
                  '                phones.append("sp")\n\n            durations.append(')
    # 2. utf-8 for Telugu .lab files
    s = s.replace('with open(text_path, "r") as f:', 'with open(text_path, "r", encoding="utf-8") as f:')
    # 3. split by the manifest (our fixed train/val), not random
    s = s.replace("        random.shuffle(out)\n        out = [r for r in out if r is not None]\n",
        "        out = [r for r in out if r is not None]\n"
        "        import csv\n"
        "        split = {}\n"
        "        mpath = self.config['path']['manifest_path']\n"
        "        for row in csv.DictReader(open(mpath, encoding='utf-8')):\n"
        "            split[row['id']] = row['split']\n")
    s = s.replace("            for m in out[self.val_size :]:\n                f.write(m + \"\\n\")",
                  "            for m in out:\n                if split.get(m.split('|')[0]) == 'train':\n                    f.write(m + \"\\n\")")
    s = s.replace("            for m in out[: self.val_size]:\n                f.write(m + \"\\n\")",
                  "            for m in out:\n                if split.get(m.split('|')[0]) == 'val':\n                    f.write(m + \"\\n\")")
    open(path, "w", encoding="utf-8").write(s)


def patch_symbols_telugu(path):
    s = open(path, encoding="utf-8").read()
    s = s.replace("_silences = [\"@sp\", \"@spn\", \"@sil\"]",
                  "_silences = [\"@sp\", \"@spn\", \"@sil\"]\n"
                  "_telugu = ['@' + chr(c) for c in range(0x0C00, 0x0C80)]")
    s = s.replace("    + _silences\n)", "    + _silences\n    + _telugu\n)")
    open(path, "w", encoding="utf-8").write(s)


def patch_open_utf8(path):
    s = open(path, encoding="utf-8").read()
    s = re.sub(r'open\(([^)]*?)\)\.read\(\)', r'open(\1, encoding="utf-8").read()', s)
    open(path, "w", encoding="utf-8").write(s)


# ---------------------------------------------------------------- configs
def preprocess_yaml(lang):
    return textwrap.dedent(f"""\
        dataset: "{lang}"
        path:
          corpus_path: "/kaggle/working/seg/{lang}/raw"
          lexicon_path: "lexicon/librispeech-lexicon.txt"
          raw_path: "/kaggle/working/seg/{lang}/raw"
          preprocessed_path: "./preprocessed_data/{lang}"
          manifest_path: "{MANIFESTS[lang]}"
        preprocessing:
          val_size: 0
          text:
            text_cleaners: ["{'english_cleaners' if lang == 'English' else 'basic_cleaners'}"]
            language: "{'en' if lang == 'English' else 'te'}"
          audio: {{sampling_rate: 22050, max_wav_value: 32768.0}}
          stft: {{filter_length: 1024, hop_length: 256, win_length: 1024}}
          mel: {{n_mel_channels: 80, mel_fmin: 0, mel_fmax: 8000}}
          pitch: {{feature: "phoneme_level", normalization: True}}
          energy: {{feature: "phoneme_level", normalization: True}}
        """)


def model_yaml(lang):
    ms = "True" if lang == "English" else "False"
    return open(os.path.join(SRC, "config/LibriTTS/model.yaml")).read().replace(
        "multi_speaker: True", f"multi_speaker: {ms}")


def train_yaml(lang):
    # budget: ~5 h of data -> few thousand steps/epoch is small; 30k steps fits one Kaggle session
    return textwrap.dedent(f"""\
        path:
          ckpt_path: "./checkpoints"
          log_path: "./output/log"
          result_path: "./output/result"
        optimizer:
          batch_size: 16
          betas: [0.9, 0.98]
          eps: 0.000000001
          weight_decay: 0.0
          grad_clip_thresh: 1.0
          grad_acc_step: 1
          warm_up_step: 2000
          anneal_steps: [14000, 18000]
          anneal_rate: 0.3
        step:
          total_step: 20000
          log_step: 100
          synth_step: 1000
          val_step: 1000
          save_step: 5000
        """)


def make_lang(lang):
    out = os.path.join(DST, lang)
    shutil.copytree(SRC, out, ignore=shutil.ignore_patterns(".git", "demo", "img", "preprocessed_data", "output", "checkpoints", "__pycache__"),
                    dirs_exist_ok=True)
    patch_preprocessor(os.path.join(out, "preprocessor/preprocessor.py"))
    if lang == "Telugu":
        patch_symbols_telugu(os.path.join(out, "text/symbols.py"))
    cfg = os.path.join(out, "config_dir")
    os.makedirs(cfg, exist_ok=True)
    open(os.path.join(cfg, "preprocess.yaml"), "w", encoding="utf-8").write(preprocess_yaml(lang))
    open(os.path.join(cfg, "model.yaml"), "w").write(model_yaml(lang))
    open(os.path.join(cfg, "train.yaml"), "w").write(train_yaml(lang))
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    # TextGrids into the place the preprocessor expects
    tg_dst = os.path.join(out, "preprocessed_data", lang, "TextGrid")
    shutil.copytree(f"/kaggle/working/seg/{lang}/aligned", tg_dst, dirs_exist_ok=True)
    print("built", out)


if __name__ == "__main__":
    langs = sys.argv[3:] or ["English", "Telugu"]
    for l in langs:
        make_lang(l)
