import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from string import punctuation

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import yaml
from scipy.io import wavfile

from text import text_to_sequence
from utils.model import get_model, get_vocoder, vocoder_infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test sentences exactly as given in the assignment (E3 and T3 keep the odd spacing on purpose).
TESTS = {
    "English": [
        ("E1", "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."),
        ("E2a", "The orchestra's conduct during the ceremony was praised by the audience."),
        ("E2b", "The researchers will conduct an experiment to evaluate the proposed method."),
        ("E3", "The rhyth m of the orchestra changed unexpectedly."),
        ("E4", "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."),
    ],
    "Telugu": [
        ("T1", "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."),
        ("T2a", "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."),
        ("T2b", "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."),
        ("T3", "పులి అడ విలో నిశ్శబ్దంగా నడిచింది."),
        ("T4", "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉండండి."),
    ],
}


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            if temp[0].lower() not in lexicon:
                lexicon[temp[0].lower()] = temp[1:]
    return lexicon


_G2P, _LEX = None, None


def english_phones(text, preprocess_config):
    """Lexicon lookup, g2p_en fallback for out-of-vocabulary words, punctuation -> sp."""
    global _G2P, _LEX
    from g2p_en import G2p

    if _G2P is None:
        _G2P = G2p()
        _LEX = read_lexicon(preprocess_config["path"]["lexicon_path"])
    from en_norm import normalize_en
    text = normalize_en(text).rstrip(punctuation)
    phones = []
    for w in re.split(r"([,;.\-\?\!\s+])", text):
        if w.lower() in _LEX:
            phones += _LEX[w.lower()]
        else:
            phones += [p for p in _G2P(w) if p != " "]
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    return phones.replace("}{", " ")


def _te_words(n):
    from num2words import num2words

    w = num2words(int(n), lang="te")
    return w.replace("వేయిల", "వేల").replace("అయిదు", "ఐదు")


def expand_numbers_te(text):
    """Digits -> Telugu words (training transcripts contain only Telugu letters)."""
    text = re.sub(r"(\d+):(\d+)", lambda m: _te_words(m[1]) + " " + _te_words(m[2]), text)

    def ordinal(m):  # 12వ -> పన్నెండవ
        w = _te_words(m[1])
        return (w[:-1] if w.endswith("ు") else w) + "వ"

    text = re.sub(r"(\d+)వ(?=\s|$)", ordinal, text)
    return re.sub(r"\d+", lambda m: _te_words(m[0]), text)


def telugu_phones(text):
    """One symbol per Telugu code point (the MFA dictionary is character-level).
    Commas/full stops become a short pause (sp); spaces are not marked, matching training."""
    text = expand_numbers_te(unicodedata.normalize("NFC", text))
    phones = []
    for ch in text:
        if 0x0C00 <= ord(ch) <= 0x0C7F:
            phones.append(ch)
        elif ch in ",.;:!?" and phones and phones[-1] != "sp":
            phones.append("sp")
    while phones and phones[-1] == "sp":
        phones.pop()
    return "{" + " ".join(phones) + "}"


def pick_font(lang):
    if lang != "te":
        return
    for f in font_manager.fontManager.ttflist:
        if "telugu" in f.name.lower():
            plt.rcParams["font.family"] = f.name
            return
    print("WARNING: no Telugu font found; install with: apt-get install -y fonts-noto-core")


def plot_variance(kind, sid, labels, values, ylabel, path):
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(labels)), 3.6))
    x = np.arange(len(labels))
    if kind == "duration":
        ax.bar(x, values, color="#4c78a8")
    else:
        ax.plot(x, values, marker="o", ms=3.5, lw=1.2, color="#e45756" if kind == "pitch" else "#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_xlim(-0.8, len(labels) - 0.2)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{sid}: predicted {kind} per phoneme")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore_step", type=int, required=True)
    ap.add_argument("-p", "--preprocess_config", required=True)
    ap.add_argument("-m", "--model_config", required=True)
    ap.add_argument("-t", "--train_config", required=True)
    ap.add_argument("--speaker", default=None, help="English only: speaker group id, e.g. S012 (default: most frequent in train)")
    ap.add_argument("--out", default=None, help="Results/<Language> folder")
    args = ap.parse_args()

    pc = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    mc = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
    tc = yaml.load(open(args.train_config), Loader=yaml.FullLoader)
    configs = (pc, mc, tc)
    lang = pc["preprocessing"]["text"]["language"]
    name = "English" if lang == "en" else "Telugu"
    out = args.out or os.path.join("..", "Results", name)
    os.makedirs(os.path.join(out, "test_generated_audios"), exist_ok=True)
    for k in ("duration", "pitch", "energy"):
        os.makedirs(os.path.join(out, "variance_adaptor_plots", k), exist_ok=True)
    pick_font(lang)

    pre = pc["path"]["preprocessed_path"]
    speakers = json.load(open(os.path.join(pre, "speakers.json")))
    stats = json.load(open(os.path.join(pre, "stats.json")))
    p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
    e_mean, e_std = stats["energy"][2], stats["energy"][3]
    hop, sr = pc["preprocessing"]["stft"]["hop_length"], pc["preprocessing"]["audio"]["sampling_rate"]


    spk_key = args.speaker
    if name == "English" and spk_key is None:
        spk_key = Counter(l.split("|")[1] for l in open(os.path.join(pre, "train.txt"), encoding="utf-8")).most_common(1)[0][0]
    spk_id = speakers[spk_key] if name == "English" else 0
    print(f"language={name} speaker={spk_key} (id {spk_id}) restore_step={args.restore_step}")

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(mc, device)

    for sid, sent in TESTS[name]:
        phones = english_phones(sent, pc) if lang == "en" else telugu_phones(sent)
        seq = np.array(text_to_sequence(phones, pc["preprocessing"]["text"]["text_cleaners"]))
        labels = [t for t in phones.strip("{}").split()]
        print(f"[{sid}] {sent}\n     phones: {phones}\n     tokens kept: {len(seq)}/{len(labels)}")
        texts = torch.from_numpy(seq).long().unsqueeze(0).to(device)
        lens = torch.tensor([len(seq)]).to(device)
        spk = torch.tensor([spk_id]).long().to(device)
        with torch.no_grad():
            o = model(spk, texts, lens, len(seq))
        mel, p_pred, e_pred, d_round, mel_len = o[1], o[2], o[3], o[5], o[9]
        wav = vocoder_infer(mel.transpose(1, 2), vocoder, mc, pc, [int(mel_len[0]) * hop])[0]
        wavfile.write(os.path.join(out, "test_generated_audios", f"{sid}.wav"), sr, wav)

        dur_ms = d_round[0].cpu().numpy() * hop / sr * 1000.0
        pitch = p_pred[0].cpu().numpy() * p_std + p_mean
        energy = e_pred[0].cpu().numpy() * e_std + e_mean
        n = min(len(labels), len(dur_ms))
        base = os.path.join(out, "variance_adaptor_plots")
        plot_variance("duration", sid, labels[:n], dur_ms[:n], "duration (ms)", os.path.join(base, "duration", f"{sid}.png"))
        plot_variance("pitch", sid, labels[:n], pitch[:n], "F0 (Hz)", os.path.join(base, "pitch", f"{sid}.png"))
        plot_variance("energy", sid, labels[:n], energy[:n], "energy", os.path.join(base, "energy", f"{sid}.png"))
        print(f"     audio {len(wav) / sr:.2f}s, mean F0 {pitch[:n].mean():.1f} Hz")


if __name__ == "__main__":
    main()
