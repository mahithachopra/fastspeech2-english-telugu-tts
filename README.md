# FastSpeech 2 TTS for English and Telugu

Two separate FastSpeech 2 models, adapted from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2):

| | English | Telugu |
|---|---|---|
| Data | Svarah (Indian-accented English, `ai4bharat/Svarah`) | IndicTTS Telugu, female voice (`SPRINGLab/IndicTTS_Telugu`) |
| Speakers | multi-speaker (115 speaker groups) | single speaker |
| Input symbols | ARPAbet phonemes (MFA `english_us_arpa`) | Telugu characters (one symbol per Unicode code point) |
| Aligner | pretrained MFA English model | MFA model trained on the 6 h subset |
| Training examples | utterances cut at pauses into 2-7 s segments (4 446) | same (3 637) |
| Vocoder | HiFi-GAN "universal" (reference repo, not retrained) | same |

```
English/ , Telugu/     train.py, preprocess.py, inference.py, config.yaml, checkpoints/ (+ model code)
Results/<Language>/    training_loss.png, validation_loss.png, test_generated_audios/, variance_adaptor_plots/{duration,pitch,energy}/
Results/<Language>_compare_step5000/    same outputs from the step-5000 checkpoint, for comparison
Report/                Technical_Report.pdf, data_manifests/ (exact train/validation split)
tools/                 setup_fs2.py / setup_seg.py (build the patched language folders), segment.py (cuts utterances at pauses), prepare_telugu_mfa.py, telugu.dict, tel_train.yaml, segment_manifests/ (split of the 2-7 s segments)
subsample.py           dataset selection (5 h train / 1 h validation per language)
```

The code reads three YAML files per language from `config_dir/` (`preprocess.yaml`, `model.yaml`, `train.yaml`); `config.yaml` is the same content merged into one file for reading.

## Checkpoints
`English/checkpoints/` and `Telugu/checkpoints/` each hold steps **5000** and **20000** (model weights only, stored in float16 so each file stays under GitHub's 100 MB limit; PyTorch converts them back on load).
Step 20000 (the final step) is the official result in `Results/<Language>/`; step 5000 is in `Results/<Language>_compare_step5000/`. Validation loss is lowest around steps 2000-4000 and rises slowly afterwards (see the report).

## Reproduce

**1. Environment.** `pip install -r requirements.txt` (versions are those used for the results). Montreal Forced Aligner 3.4.3 is separate:
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba create -y -p ./mfa_env -c conda-forge python=3.11 montreal-forced-aligner
```
Vocoder: copy `hifigan/generator_universal.pth.tar.zip` from the reference repo into `English/hifigan/` and `Telugu/hifigan/` and unzip it there.

**2. Data subset.** `python subsample.py` (needs a Hugging Face token with access to the gated Svarah dataset). Writes `raw_data/English`, `raw_data/Telugu` as 22 050 Hz wav + `.lab` pairs and the two manifests in `Report/data_manifests/` (the split is fixed by these files).
- English: utterances 1-15 s; whole speaker groups go to validation (14 groups, ~1.0 h) so no speaker appears in both sets; training is drawn from the other groups (~5.0 h, 101 groups).
- Telugu: female speaker only, utterances 1.5-20 s, duplicate sentences removed, random split (~5.0 h / ~1.0 h).

**3. Alignment (MFA).**
```bash
# English: pretrained model, g2p for out-of-dictionary words
mfa model download acoustic english_us_arpa; mfa model download dictionary english_us_arpa; mfa model download g2p english_us_arpa
mfa align raw_data/English english_us_arpa english_us_arpa aligned/English --g2p_model_path english_us_arpa --beam 100 --retry_beam 400 -j 4 --clean

# Telugu: clean transcripts, character dictionary, train an aligner, then align
python tools/prepare_telugu_mfa.py raw_data/Telugu telugu.dict
mfa train raw_data/Telugu telugu.dict aligned/Telugu --config_path tools/tel_train.yaml --single_speaker -j 3 --clean
```
MFA 3.4.3 finishes `mfa train` but then crashes while writing its model card for a single-speaker corpus (`Undefined.__format__` in `corpus_template.md`) before exporting TextGrids. The trained model is kept in `~/Documents/MFA/Telugu/triphone/log/acoustic_model.zip`; copy it out and run
`mfa align raw_data/Telugu telugu.dict telugu_acoustic.zip aligned/Telugu --single_speaker --beam 100 --retry_beam 400 -j 3 --clean`.

**3b. Segmentation.** Long utterances (Telugu median 18 s) are cut at pauses of at least 80 ms into 2-7 s segments, reusing the MFA TextGrids (no re-alignment). Segments inherit the train/validation split of their parent utterance, so nothing overlaps.
```bash
python tools/segment.py raw_data/English aligned/English Report/data_manifests/English_manifest.csv seg/English English
python tools/segment.py raw_data/Telugu  aligned/Telugu  Report/data_manifests/Telugu_manifest.csv  seg/Telugu  Telugu
```
This gives 4 524 English segments (4.98 h train / 1.02 h val before trimming) and 3 637 Telugu segments (4.97 h / 0.99 h). The segment manifests are in `tools/segment_manifests/`.

**4. Build the language folders, preprocess, train** (paths in `tools/setup_fs2.py` and `tools/setup_seg.py` are the Kaggle ones; edit the constants at the top):
```bash
git clone https://github.com/ming024/FastSpeech2.git
python tools/setup_fs2.py FastSpeech2 project English Telugu      # copies + patches the reference code, writes config_dir/
python tools/setup_seg.py project/English project2 English        # same folders, pointed at the segmented data, 20k-step schedule
python tools/setup_seg.py project/Telugu  project2 Telugu
cd project2/English        # or project2/Telugu
python preprocess.py config_dir/preprocess.yaml
python train.py -p config_dir/preprocess.yaml -m config_dir/model.yaml -t config_dir/train.yaml
# if a session stops, resume with: python train.py --restore_step <last saved step> -p ... -m ... -t ...
```
Training: 20 000 steps, batch 16, warm-up 2000, learning-rate decay at steps 14 000 and 18 000, a checkpoint every 5000 steps; about 2.5 h (English) and 2 h (Telugu) on one T4 each (both were stopped by the notebook session at step 15 000-16 000 and resumed from the step-15 000 checkpoint).

**5. Inference and plots**
```bash
cd English                # or Telugu
python inference.py --restore_step 20000 -p config_dir/preprocess.yaml -m config_dir/model.yaml -t config_dir/train.yaml
python plot_losses.py --log logs --out ../Results/English --title English
```
`inference.py` synthesises the five test sentences of each language (text is fixed in the script, kept exactly as given, including the odd spacing in E3/T3) and writes the wav files and the predicted duration / pitch / energy plots under `../Results/<Language>/`.

## Changes to the reference implementation

Shared fixes (needed for current library versions and for this data):
- `audio/stft.py`, `audio/audio_processing.py`: keyword arguments for `librosa.util.pad_center` and `librosa.filters.mel` (librosa 0.11).
- `utils/tools.py`: energy targets cast to float32 (float64 broke the backward pass).
- `utils/model.py`: `torch.load(..., weights_only=False)` (PyTorch 2.6+ default).
- `preprocessor/preprocessor.py`: MFA 3 writes silence as an empty label, mapped to `sp`; transcripts read as UTF-8; **train/validation split taken from the manifests** instead of a random shuffle.
- `inference.py` (new): frontend, synthesis, variance-adaptor plots. `plot_losses.py` (new): TensorBoard loss curves.

English: multi-speaker with 115 speaker embeddings; `g2p_en` for out-of-lexicon words; `en_norm.py` expands times, years and numbers into words before g2p (the g2p model cannot read digits).

Telugu (new language support):
- `text/symbols.py`: adds the Telugu Unicode block U+0C00-U+0C7F as symbols.
- Character-level pronunciation dictionary and an MFA acoustic model trained on the subset (no pretrained Telugu aligner exists).
- Inference frontend: one symbol per code point, punctuation becomes a short pause, digits are expanded into Telugu words (`num2words`, e.g. 2026 -> రెండు వేల ఇరవై ఆరు, 12వ -> పన్నెండవ, 10:45 -> పది నలభై ఐదు) and re-normalised to NFC.
- Segmentation (`tools/segment.py`): the first run trained on whole utterances (Telugu clips up to 20 s, which needed `max_seq_len` 1800 because the decoder silently truncates longer targets); the final run uses 2-7 s segments (none longer than 11 s), so `max_seq_len` stays at 1000.

## Limitations (details in the report)
- 4.59 h (English, 3 695 train / 751 validation segments) and 3.92 h (Telugu, 3 030 / 607) remain after silence trimming and alignment drops in total (the 5 h / 1 h selection is made before trimming).
- The generated speech is only partly intelligible; with about 4 h of training audio per language the models overfit early. See the report for the per-sentence analysis. A first run without segmentation (30 000 steps) was worse and is described in the report as a baseline.
- Svarah is an ASR benchmark (varied recording conditions, read and spontaneous speech); IndicTTS audio is 16 kHz, upsampled to 22 050 Hz for the vocoder.
- Both models overfit early (see validation curves); speaker groups for English are inferred from metadata (115 groups vs. 117 speakers in the paper).