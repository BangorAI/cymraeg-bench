# CymraegBench Voice v0.1

Mae'r estyniad llais yn meincnodi adnabod lleferydd (ASR) a synthesis
testun-i-leferydd (TTS) heb rwymo'r harnais i un framework. Mae pob model yn
addasydd gorchymyn mewn proses ar wahân, felly gall sherpa-onnx, Vosk/Kaldi,
Transformers a Piper ddefnyddio amgylcheddau annibynnol.

## Protocol ASR

- Defnyddio'r un WAV mono ar gyfer pob model a chadw'r drefn yn benderfynedig.
- Normaleiddio Unicode NFC, priflythrennau, atalnodi ac apostroffau cyn sgorio.
- Cyfrifo WER a CER micro: cyfanswm mewnosodiadau, dileadau ac amnewidiadau
  wedi'i rannu â chyfanswm yr unedau cyfeirio.
- Cofnodi'r prediction crai a normaleiddiedig, latency, real-time factor,
  revision, trwydded a metadata megis acen.
- Peidio â hyfforddi ar unrhyw achos yn y set prawf.

Mae catalog v0.1 yn cynnwys addaswyr ar gyfer BangorAI Zipformer,
`techiaith/whisper-large-v3-ft-verbatim-cy-en`, Wav2Vec2 Techiaith a
Kaldi/Vosk Techiaith. Mae'r SHA Hugging Face wedi'i binio ar gyfer pob model.

## Protocol TTS

Mae 30 prompt CC0 yn cwmpasu sgwrs, lleoedd, rhifau, treigladau, talfyriadau,
termau technegol a chyfnewid cod. Mae'r harnais yn cofnodi latency, RTF,
sample rate, clipping, distawrwydd a methiannau. Ar gyfer safon y llais rhaid
defnyddio prawf gwrando dall, wedi'i hapdrefnu, gyda sgoriau 1–5 ar gyfer:

1. dealladwyedd;
2. naturioldeb;
3. ynganu'r eitem darged;
4. rhythm a phwyslais.

Ni ddylid cyfuno'r sgoriau dynol â WER mewn un rhif cyffredinol.

## Rhedeg

```bash
# Nid yw'r sain prawf yn Git. Lawrlwythwch parquet ARFOR pinned:
mkdir -p downloads/arfor
curl --fail --location --continue-at - \
  --output downloads/arfor/test_clean-00000-of-00001.parquet \
  'https://huggingface.co/datasets/cymen-arfor/lleisiau-arfor/resolve/0665ea3e755d9864985344512b7d346363b9b806/data/test_clean-00000-of-00001.parquet?download=true'
uv run --locked python scripts/prepare_voice_arfor.py \
  --parquet downloads/arfor/test_clean-00000-of-00001.parquet

uv run --locked cymraeg-bench voice validate

uv run --locked cymraeg-bench voice run \
  --model-ids bangorai-zipformer-cy,techiaith-whisper-large-v3-verbatim \
  --suite-ids arfor-test-clean-v0.1 \
  --output runs/voice-v0.1.jsonl

uv run --locked cymraeg-bench voice report runs/voice-v0.1.jsonl \
  --markdown results/voice-v0.1/leaderboard.md \
  --csv results/voice-v0.1/leaderboard.csv

# Ar ôl rhediad TTS, creu pecyn dall ac allwedd ar wahân:
uv run --locked cymraeg-bench voice listening-pack runs/voice-v0.1.jsonl \
  --output-dir runs/voice-v0.1-listening --seed 1

uv run --locked cymraeg-bench voice listening-report \
  --ratings runs/sgoriau-gwrandawr-01.csv runs/sgoriau-gwrandawr-02.csv \
  --key runs/voice-v0.1-listening/allwedd-breifat.jsonl \
  --markdown results/voice-v0.1/gwrando.md \
  --csv results/voice-v0.1/gwrando.csv
```

Mae `voice run` yn checkpointio pob achos i JSONL ac yn hepgor rhesi
llwyddiannus wrth ailddechrau. Nid yw modelau mawr wedi'u galluogi'n ddiofyn;
dewiswch IDs yn benodol ar ôl paratoi eu hamgylcheddau a'r newidynnau llwybr
yn `config/voice-models.toml`.
