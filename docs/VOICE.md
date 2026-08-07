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

Mae catalog v0.1 yn cynnwys BangorAI Zipformer a phob un o'r 19 repo a
ddychwelwyd gan API Hugging Face ar gyfer
`author=techiaith&filter=automatic-speech-recognition` ar 7 Awst 2026. Mae'r
snapshot cyflawn yn `config/techiaith-asr-catalog.json`; mae prawf awtomatig yn
methu os nad oes cofnod benchmark â'r un SHA ar gyfer pob repo. Mae'r runtime
priodol wedi'i nodi ar gyfer Transformers, CTranslate2, whisper.cpp a
Vosk/Kaldi. Mae hefyd yn cynnwys `DewiBrynJones/kaldi-cy-2606` fel comparator
ymchwil ar wahân; nid yw'r repo hwnnw'n datgan trwydded, felly ni ddylid
ailddosbarthu na defnyddio'r pwysau mewn cynnyrch heb eglurhad.
Mae'r addaswyr ASR swyddogol yn defnyddio protocol JSONL hirhoedlog: caiff y
model ei lwytho unwaith, yna mesurir pob clip heb gynnwys cost cychwyn y model
yn y latency na'r RTF.
Mae addasydd sherpa-onnx hefyd yn derbyn `--decoding-method` a
`--max-active-paths`, fel bod greedy a modified beam search yn gallu cael eu
cymharu heb newid y pwysau.

Mae resumability yn cynnwys revision y model yn yr allwedd checkpoint. Felly,
os yw ID rhesymegol yn aros yr un fath ond bod y SHA yn newid, caiff pob achos
ei ail-redeg ac mae'r adroddiad yn cadw'r ddau revision ar wahân. Wrth ail-redeg
achos a fethodd, dim ond y cofnod diweddaraf ar gyfer yr un
model/revision/set/achos sy'n cyfrannu at y leaderboard.

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
Mae'r catalog yn trin tri speaker pro `cy_GB-bu_tts` (benyw Gogledd, benyw De,
a gwryw Gogledd) fel amrywiadau ar wahân er eu bod yn rhannu'r un model 77 MB.

## Rhedeg

```bash
# Gosodwch build PyTorch CPU/CUDA sy'n addas i'r peiriant yn gyntaf. Yna:
python -m pip install -e '.[voice-asr,voice-tts]'

# Nid yw'r sain prawf yn Git. Lawrlwythwch parquet ARFOR pinned:
mkdir -p downloads/arfor
curl --fail --location --continue-at - \
  --output downloads/arfor/test_clean-00000-of-00001.parquet \
  'https://huggingface.co/datasets/cymen-arfor/lleisiau-arfor/resolve/0665ea3e755d9864985344512b7d346363b9b806/data/test_clean-00000-of-00001.parquet?download=true'
uv run --locked python scripts/prepare_voice_arfor.py \
  --parquet downloads/arfor/test_clean-00000-of-00001.parquet

uv run --locked cymraeg-bench voice validate

uv run --locked cymraeg-bench voice run \
  --model-ids bangorai-zipformer-cy,techiaith-whisper-large-cy-en-2607 \
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
yn `config/voice-models.toml`. Mae extras `voice-asr` a `voice-tts` wedi'u
pinio; cedwir PyTorch y tu allan i'r extras oherwydd bod ei build CUDA/CPU yn
dibynnu ar y peiriant.

Mae'r modelau CTranslate2 yn lawrlwytho snapshot wedi'i binio drwy
`huggingface_hub` ac yn defnyddio `faster-whisper`. Ar gyfer y tri artifact
whisper.cpp, gosodwch `WHISPER_CPP_CLI` i'r executable `whisper-cli` a'r tri
newidyn `TECHIAITH_WHISPER_BASE_*_MODEL` i'w ffeiliau `ggml-model.bin`. Ar gyfer
Kaldi, gosodwch `TECHIAITH_KALDI_DIR` a `TECHIAITH_KALDI_2601_DIR` i'r
cyfeiriaduron model sydd wedi'u dadbacio. Nid oes angen y newidynnau hyn oni bai
bod yr IDs perthnasol wedi'u dewis.
