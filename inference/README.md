# Modelau lleol ar yr RTX 4090

Mae'r cyfeiriadur hwn yn gweini Caernarfon 3B, Mwydryn Phi-2, Mistral 7B
Cymraeg v2, Mwydryn 7B v2 a Techiaith Llama 3.2 1B Welsh SFT drwy API sy'n
gydnaws ag OpenAI. Mae Caernarfon a Mwydryn Phi-2 yn defnyddio hyd cyd-destun
o 2,048 tocyn; defnyddir 4,096 ar gyfer y modelau eraill. Defnyddir had 1 a
fformat sgwrsio penodol er mwyn i'r canlyniadau fod yn ailadroddadwy.

Mae'r amgylchedd inference wedi'i greu ar wahân i amgylchedd yr harnais:

```bash
uv venv --python 3.12 inference/.venv
uv pip install --python inference/.venv/bin/python vllm ninja
```

Nid model cyfarwyddyd yw `britllm/britllm-3b-v0.1`, ac nid oes ganddo dempled
sgwrsio brodorol. Mae `templates/caernarfon-chat.jinja` felly'n fformatio'r
cyfarwyddyd fel testun plaen gyda `Defnyddiwr:` a `Cynorthwyydd:`. Cafodd
Mwydryn ei fireinio ar ddata yn null Alpaca, felly mae ei dempled yn defnyddio
penawdau `### Instruction:` ac `### Response:`. Rhaid cofnodi'r ddau addasydd
prompt wrth gyhoeddi sgoriau.

Mae `BangorAI/Mistral-7B-Cymraeg-Welsh-v2` yn fodel cyfarwyddyd dwyieithog.
Defnyddir union fformat Llama 2 a nodir ar gerdyn y model, gan gynnwys y
cyfarwyddyd system Cymraeg. Caiff y model hwn ei weini ar ei ben ei hun mewn
`bfloat16`, gan fod angen y rhan fwyaf o gof yr RTX 4090 arno.

Mae `BangorAI/mwydryn-7b-fersiwn-2` hefyd yn seiliedig ar Mistral 7B ac yn
defnyddio'r un fformat Llama 2. Caiff ei weini ar wahân mewn `bfloat16` ar
borth 8006.

Mae repo `techiaith/llama-3.2-1b-welsh-sft` yn cynnwys addasydd LoRA yn y
gwraidd a model BF16 llawn o dan `sft/`. Mae'r sgript yn lawrlwytho'r
is-gyfeiriadur llawn yn unig ac yn ei weini ar borth 8007. Mae demo swyddogol
Techiaith yn nodi bod yr hyfforddiant yn defnyddio fformat Alpaca gyda'r
rhagymadrodd Saesneg penodol sydd yn
`templates/techiaith-llama-sft-chat.jinja`; defnyddir yr un fformat yma. Caiff
y cyfarwyddyd system a'r cwestiwn eu cyfuno yn un maes `Instruction` fel nad
yw cyfyngiadau ateb yr harnais yn cael eu colli.
Mae'r catalog hefyd yn gosod `repetition_penalty = 1.15`, yr un gwerth ag ap
swyddogol Techiaith; hebddo, mae'r model 1B yn tueddu i barhau hyd at y terfyn
allbwn hyd yn oed ar gwestiynau byr.

Mae Mwydryn yn cynnwys cod Phi-2 o Transformers 4.36.2 ac nid yw ei enwau
pwysau'n gydnaws â llwythwr Phi modern vLLM. Mae `mwydryn_server.py` felly'n
darparu'r un endpoint OpenAI drwy'r union fersiwn Transformers a gofnodwyd yng
nghyfluniad y model. Cedwir y pecynnau cydnaws yn `mwydryn-compat/`; mae Torch,
FastAPI ac Uvicorn yn cael eu hailddefnyddio o amgylchedd vLLM.

I ail-greu'r haen gydnaws:

```bash
uv pip install --target inference/mwydryn-compat \
  'transformers==4.36.2' 'einops>=0.7,<1'
```

## Cychwyn a phrofi

```bash
cd ~/aisteddfod
inference/start.sh caernarfon
inference/start.sh mwydryn
inference/start.sh mistral-cymraeg
inference/start.sh mwydryn-7b
inference/start.sh techiaith-llama-sft
inference/smoke-test.sh caernarfon
inference/smoke-test.sh mwydryn
inference/smoke-test.sh mistral-cymraeg
inference/smoke-test.sh mwydryn-7b
inference/smoke-test.sh techiaith-llama-sft
```

Mae'r gweinyddion yn gwrando ar `127.0.0.1:8002`, `127.0.0.1:8003`,
`127.0.0.1:8005`, `127.0.0.1:8006` a `127.0.0.1:8007` yn unig.
Defnyddiwch `inference/stop.sh all` i'w stopio. Mae'r logiau o dan
`inference/logs/` a'r PIDau o dan `inference/run/`. Gall y lawrlwythiad cyntaf
gymryd peth amser; mae `start.sh` yn aros hyd at 30 munud yn ddiofyn. Gosodwch
`AISTEDDFOD_START_TIMEOUT` i newid hynny.

Os na fydd y ddau fodel yn ffitio ar y GPU ar yr un pryd, rhedwch nhw fesul un
gan ddefnyddio'r un gorchmynion. Gellir rhoi mwy o gof i un gweinydd drwy
osod `CAERNARFON_GPU_MEMORY=0.85`, `MWYDRYN_GPU_MEMORY=0.85`,
`MISTRAL_CYMRAEG_GPU_MEMORY=0.90`, `MWYDRYN_7B_GPU_MEMORY=0.90` neu
`TECHIAITH_LLAMA_SFT_GPU_MEMORY=0.85` cyn ei gychwyn.

## Rhedeg yr harnais

Prawf mwg un achos o bob set:

```bash
uv run --locked cymraeg-bench run \
  --run-id lleol-mwg-001 \
  --models caernarfon-3b,mwydryn,bangor-mistral-cymraeg-v2,mwydryn-7b-v2,techiaith-llama-3-2-1b-sft \
  --max-cases 1 --seed 1 --workers 2
```

Rhediad cyhoeddi CymraegBench v0.1:

```bash
uv run --locked cymraeg-bench run \
  --run-id cymraegbench-v0.1-lleol \
  --models caernarfon-3b,mwydryn,bangor-mistral-cymraeg-v2,mwydryn-7b-v2,techiaith-llama-3-2-1b-sft \
  --max-cases 100 --seed 1 --workers 2
```

Archwiliad Iaith CCC:

```bash
uv run --locked cymraeg-bench run \
  --run-id ccc-v0.1-lleol \
  --models caernarfon-3b,mwydryn,bangor-mistral-cymraeg-v2,mwydryn-7b-v2,techiaith-llama-3-2-1b-sft \
  --suites ccc-cywiro-esbonio,ccc-priod-ddull,ccc-termau,ccc-crynhoi-dyfynnu,ccc-cyfeiriadau,ccc-cyfarwyddyd-cymraeg,ccc-cyfarwyddyd-saesneg \
  --seed 1 --workers 2
```

Mae `uv run --locked cymraeg-bench report runs/<run-id>.sqlite3` yn creu'r
crynodeb arferol; defnyddiwch `ccc-report` ar gyfer dimensiynau CCC.

## Y pum model, yn awtomatig

Mae'r rhedwr canlynol yn newid y GPU rhwng y pum model, yn ailddechrau unrhyw
rhediad rhannol sydd â'r un ID, ac yn creu'r ddau fath o adroddiad:

```bash
inference/run-local-benchmarks.sh
```

Yn ddiofyn, defnyddir `cymraegbench-v0.1-lleol` fel rhagddodiad yr ID a hyd at
100 achos fesul set. Rhoddir hyd at dri chynnig i achosion sydd â gwall
trosglwyddo neu derfyn allbwn; cedwir gwall parhaol yn y canlyniad a symudir
ymlaen at y model nesaf. Gellir newid y gosodiadau heb olygu'r sgript:

```bash
LOCAL_BENCH_PREFIX=lleol-ailadrodd-001 \
LOCAL_BENCH_MAX_CASES=3 \
LOCAL_BENCH_MAX_ATTEMPTS=2 \
inference/run-local-benchmarks.sh
```

I redeg neu ailddechrau is-set yn unig, rhowch IDau'r catalog mewn rhestr
atalnodau:

```bash
LOCAL_BENCH_MODELS=techiaith-llama-3-2-1b-sft \
inference/run-local-benchmarks.sh
```
