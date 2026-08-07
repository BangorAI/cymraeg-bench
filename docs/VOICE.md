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
Mae token gorchymyn llythrennol `python` yn cael ei ddatrys i `sys.executable`,
fel bod yr adapter yn defnyddio'r un interpreter a dependency environment â'r
harnais. Gellir rhoi llwybr interpreter absoliwt yn y catalog pan fo angen
amgylchedd ar wahân.
Mae addasydd sherpa-onnx hefyd yn derbyn `--decoding-method` a
`--max-active-paths`, fel bod greedy a modified beam search yn gallu cael eu
cymharu heb newid y pwysau.
Heb flags, mae'n darllen `CYMRAEG_ZIPFORMER_DECODING_METHOD` a
`CYMRAEG_ZIPFORMER_MAX_ACTIVE_PATHS`; mae'r finalizer benchmark yn eu gosod o'r
dev lock fel bod y CLI a'r porwr yn defnyddio'r un decoder.

Mae resumability yn cynnwys revision y model yn yr allwedd checkpoint. Felly,
os yw ID rhesymegol yn aros yr un fath ond bod y SHA yn newid, caiff pob achos
ei ail-redeg ac mae'r adroddiad yn cadw'r ddau revision ar wahân. Wrth ail-redeg
achos a fethodd, dim ond y cofnod diweddaraf ar gyfer yr un
model/revision/set/achos sy'n cyfrannu at y leaderboard.
Mae'r addasydd Transformers yn datrys y snapshot SHA cyfan i lwybr lleol cyn
creu'r pipeline. Mae hyn yn pinio tokenizer a KenLM hefyd; nid yw'n dibynnu ar
ymddygiad `pyctcdecode` sy'n gallu lawrlwytho'r language model o `main` hyd yn
oed pan fo `revision` y model acwstig wedi'i osod.
Yn revision piniedig
`techiaith/wav2vec2-xlsr-53-ft-cy-en-withlm@3d462e5b85490b9e16438ed524bc3c0deed2485f`,
mae'r `alphabet.json` cyhoeddedig yn cynnwys 73 label ond mae pen CTC y model
yn allbynnu 46 logit. Mae'r addasydd yn canfod yr anghysondeb dimensiwn ac yn
ail-greu'r alphabet o `vocab.json` a blank ID y model, gan gadw `lm.bin`,
unigrams a pharamedrau KenLM o'r un snapshot yn union. Nid yw'n defnyddio
tokenizer, LM na phwysau o revision arall.

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
`huggingface_hub` ac yn defnyddio `faster-whisper`. Ar CUDA defnyddir FP16 fel
nad yw quantization yn gwanhau comparator Techiaith; defnyddir INT8 ar CPU yn
unig. Gellir gosod `VOICE_BENCH_CT2_COMPUTE_TYPE` yn benodol ar gyfer smoke
test diagnostig, ond mae'r release benchmark 4090 yn cadw'r default FP16.
Ar gyfer y tri artifact
whisper.cpp, gosodwch `WHISPER_CPP_CLI` i'r executable `whisper-cli` a'r tri
newidyn `TECHIAITH_WHISPER_BASE_*_MODEL` i'w ffeiliau `ggml-model.bin`. Ar gyfer
Kaldi, gosodwch `TECHIAITH_KALDI_DIR` a `TECHIAITH_KALDI_2601_DIR` i'r
cyfeiriaduron model sydd wedi'u dadbacio. Nid oes angen y newidynnau hyn oni bai
bod yr IDs perthnasol wedi'u dewis.

Ar y peiriant benchmark Linux/CUDA, mae'r ddwy sgript ganlynol yn adeiladu
whisper.cpp ar y commit yn `config/voice-runtimes.env`, yn lawrlwytho'r tri
`ggml-model.bin` a'r ddau archive Kaldi wrth eu SHA, ac yn creu ffeil env:

```bash
./scripts/bootstrap_whisper_cpp.sh
python scripts/prepare_techiaith_asr_assets.py
source models/techiaith/env.sh
```

Mae extraction y tar Kaldi yn gwrthod path traversal, links a device nodes.
Cedwir `assets.json` ochr yn ochr â'r models i gofnodi SHA a llwybr pob asset;
mae `models/`, `downloads/`, `vendor/` a'r dependency targets lleol wedi'u
hepgor o Git.

Ar ôl i'r rheolydd hyfforddi gloi'r checkpoint a'r decoder gorau ar y set dev,
gellir gadael y goruchwyliwr release yn rhedeg. Mae'n gwirio mynediad i bob
artifact Techiaith pinned cyn cyffwrdd â'r manifest test, yn aros am y bundle
ONNX/WASM, yn creu runtime Python o'r `uv.lock`, yn paratoi'r 19 model, yn
ail-greu'r manifest ARFOR 3,445 achos, ac yn rhedeg un model ar y tro. Mae'r
JSONL yn checkpointio pob achos, felly gellir ailgychwyn yr un gorchymyn:

```bash
python scripts/run_release_asr_benchmark.py \
  --release-status /path/to/exp/release-finalization-status.json \
  --zipformer-root /path/to/zipformer-cymraeg \
  --arfor-parquet /path/to/test_clean-00000-of-00001.parquet \
  --dewi-model-dir /path/to/dewi-kaldi-2606 \
  --python /path/to/zipformer-cymraeg/.venv/bin/python
```

Mae'r runner yn mynnu `uv 0.5.29` yn union. Mae'n chwilio yn gyntaf am `--uv`,
wedyn am `uv` wrth ymyl y `--python` penodol, ac yn olaf ar `PATH`; mae hyn yn
atal environment SSH gwahanol rhag defnyddio fersiwn export anghydnaws. Mae'r
extra Linux hefyd yn pinio `sherpa-onnx-core==1.13.4` yn uniongyrchol fel bod ei
wheel a'i SHA-256 yn bresennol pan osodir gyda `pip --require-hashes`.

Ni fydd y gate yn pasio oni bai bod BangorAI wedi cwblhau 3,445/3,445 ac â WER
is na'r gorau o bob un o'r 19 model Techiaith ar yr un manifest. Cedwir statws
peiriant-ddarllenadwy yn `runs/voice-v0.1-status.json` a'r leaderboard o dan
`results/voice-v0.1/`. Mae revision rhes BangorAI yn fingerprint SHA-256 o'r
union bundle WASM terfynol, felly ni all canlyniad hen checkpoint basio gate
release newydd. Os bydd adapter neu achos yn methu dros dro, mae'r runner yn
ailafael hyd at dair gwaith yn ddiofyn (`--model-attempts`), gan ail-redeg dim
ond yr achosion coll neu aflwyddiannus; mae'n dal i fethu'r release os nad yw
pob un o'r 3,445 achos yn llwyddo.
