# CymraegBench

**[Cymraeg](#cymraeg) · [English](#english)**

<a id="cymraeg"></a>

## Cymraeg

Meincnod agored ac ailadroddadwy ar gyfer gwerthuso modelau iaith yn y
Gymraeg. CymraegBench yw'r harnais ymchwil sy'n sail i sgorfwrdd
[AIsteddfod](https://aisteddfod.com/); mae'n cyfuno setiau BritEval, Techiaith,
Cardiff NLP a phrotocolau BangorAI mewn un broses dryloyw.

Mae hefyd yn cynnwys [Archwiliad Iaith CCC](data/ccc-audit/README.md): 232 o
achosion agored sy'n troi canfyddiadau ansoddol adroddiad Kara Lewis am AI,
y Gymraeg ac addysg uwch yn saith is-set ailadroddadwy. Mae'r archwiliad yn
mesur cywiro ac esbonio ar wahân, priod-ddull, termau, crynhoi, dyfyniadau,
cyfeiriadau ac effaith iaith y cyfarwyddyd.

Mae [CymraegBench Voice v0.1](docs/VOICE.md) yn ychwanegu protocol ASR a TTS:
WER/CER, latency a real-time factor ar gyfer trawsgrifio, ynghyd â phecyn
promptiau CC0 a phrawf gwrando dall ar gyfer synthesis. Mae'r catalog pinned
yn cynnwys Zipformer BangorAI a modelau Whisper, Wav2Vec2, Kaldi a Piper
Techiaith.

Gweler [PLAN.md](PLAN.md) am y protocol, tarddiad y setiau a'r rhesymau dros y
dewisiadau sgorio. Nid yw canlyniadau rhediadau mwg yn sgoriau cyhoeddadwy:
eu diben yw gwirio'r harnais, terfynau tocynnau a fformat atebion cyn rhediad
swyddogol.

### Cychwyn

```bash
uv sync --locked
cp .env.example .env
# ychwanegu'r tair allwedd API at .env
uv run --locked --env-file .env cymraeg-bench validate
uv run --locked --env-file .env cymraeg-bench plan --max-cases 3
uv run --locked --env-file .env cymraeg-bench run --max-cases 3 --max-usd 5
uv run --locked --env-file .env cymraeg-bench report runs/<run-id>.sqlite3
uv run --locked --env-file .env cymraeg-bench ccc-report runs/<run-id>.sqlite3
```

Mae `uv` yn rheoli Python 3.11, `.venv`, y dibyniaethau a `uv.lock`. Mae
`--locked` yn gwrthod rhedeg os nad yw'r lockfile yn cyfateb i
`pyproject.toml`; mae `--env-file .env` yn llwytho'r allweddi heb eu cadw yn y
gronfa ganlyniadau. Gellir defnyddio rheolwr cyfrinachau yn lle `.env`.

### Llwybro

- `OPENAI_API_KEY`: GPT-5.6 Sol drwy `POST /v1/responses`.
- `ANTHROPIC_API_KEY`: Claude Opus 4.8, Claude Opus 5 a Claude Fable 5 drwy
  `POST /v1/messages`.
- `OPENROUTER_API_KEY`: Grok, Kimi, Muse Spark 1.2 (pwysau agored), GLM a DeepSeek drwy
  `POST /api/v1/chat/completions`.
- Modelau lleol: cychwyn vLLM/TGI/Ollama â rhyngwyneb Chat Completions a
  gosod yr `*_API_BASE` perthnasol. Wedyn troi `enabled = true` ymlaen yn
  `config/models.toml`.

Mae `validate --check-remote` yn cymharu IDs OpenRouter â'r catalog byw heb
anfon prompt at fodel. Nid yw `plan` yn gwneud unrhyw alwad fodel na gwario
credyd. Mae `run` yn gwrthod dechrau os oes allwedd ofynnol ar goll.

### Data preifat

Rhowch themâu'r gynghanedd yn:

```text
data/private/cynghanedd-themes.jsonl
{"id":"c001","theme":"..."}
```

Rhowch ddata trwyddedig Barddas yn `data/private/barddas.jsonl`. Mae'r ddau
lwybr wedi'u hanwybyddu gan Git. Gosodwch `CYNGHANEDD_SCORER_COMMAND` neu
`BARDDAS_SCORER_COMMAND` i raglen sy'n darllen JSON o stdin ac yn dychwelyd
JSON. Nid yw'r prosiect yn dyblygu cynnwys cudd na thrwyddedig.

### Allbynnau

Mae'r canlyniadau cyhoeddus yn cynnwys y
[sgorfwrdd saith-model gwreiddiol](results/cymraegbench-v0.1/leaderboard.md)
a'r [sgorfwrdd estynedig 15-model](results/cymraegbench-v0.1-estynedig/README.md).
Mae'r fersiwn estynedig yn cynnwys rhediadau set lawn Jupiter-N-120B a
Chaernarfon 3B Cymraeg Instruct v0.1, yn ogystal â'r samplau blaenorol.
Mae [rhediad Claude Opus 5](results/cymraegbench-claude-opus-5-v0.1/leaderboard.md)
yn ychwanegu'r model newydd ar yr un sampl penderfynedig, ynghyd ag
Archwiliad Iaith CCC.
Mae [rhediad Muse Spark 1.2](results/cymraegbench-muse-spark-1-2-v0.1/leaderboard.md)
yn defnyddio'r un sampl a'r un sgorwyr drwy OpenRouter.

Mae pob rhediad yn creu `runs/<run-id>.sqlite3`; gall `report` greu Markdown a
CSV. Mae'r gronfa'n cadw'r prompt, ateb, cyfeirnod, sgôr, tocynnau, cost,
latency, model gwirioneddol a metadata darparwr. Nid yw'n cadw allweddi API.

Nid yw'r storfa Git hon yn ailddosbarthu'r setiau trydydd parti. Mae'r
addaswyr yn eu lawrlwytho o'u ffynonellau gwreiddiol ac yn pinio'r fersiwn lle
bo modd; mae telerau a thrwydded pob set yn parhau'n gymwys.

### Datblygu

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked cymraeg-bench validate
uv run --locked cymraeg-bench plan --max-cases 1
```

Ar ôl newid dibyniaeth, defnyddiwch `uv add`, `uv remove` neu `uv lock`, ac
yna cadwch `pyproject.toml` ac `uv.lock` gyda'i gilydd.

Cedwir `aisteddfod-bench` fel alias cydnaws ar gyfer sgriptiau presennol.

Ar gyfer rhediad swyddogol, gellir defnyddio `--workers 7` i redeg hyd at un
galwad yr un pryd ar gyfer pob model. Mae pob ysgrifen SQLite yn dal i
ddigwydd yn y brif broses, ac mae'r checkpoint yn ailgeisio rhesi gwall yn
unig wrth ailddechrau'r un `--run-id`.

Protocol cyhoeddi CymraegBench v0.1 yw hyd at 100 achos penderfynedig o bob
set, gyda'r had 1 a therfyn cost diogel:

```bash
uv run --locked --env-file .env cymraeg-bench run \
  --run-id cymraegbench-v0.1 \
  --max-cases 100 --seed 1 --workers 7 --max-usd 25
```

Pan fydd pob galwad wedi cwblhau heb wall, cynhyrchir y sgorfwrdd a'r metadata
cyhoeddadwy fel hyn:

```bash
uv run --locked cymraeg-bench leaderboard runs/cymraegbench-v0.1.sqlite3
```

Mae'r sgôr cyffredinol yn rhoi pwysau cyfartal i gymedr macro naw prawf rhesymu
Cymraeg a chymedr macro naw prawf Cymraeg ymarferol. Dangosir BLEU y prawf
cyfieithu deddfwriaeth ar wahân. Os bydd galwad yn methu, ailgeisir y rhes
honno'n unig; mae `metadata.json` yn cadw cyfluniad dechrau'r rhediad a'r
cyfluniad terfynol a ddefnyddiwyd wrth ailgeisio.

I redeg yr archwiliad CCC yn unig:

```bash
uv run --locked --env-file .env cymraeg-bench run \
  --run-id ccc-v0.1-beta --suites ccc-cywiro-esbonio,ccc-priod-ddull,ccc-termau,ccc-crynhoi-dyfynnu,ccc-cyfeiriadau,ccc-cyfarwyddyd-cymraeg,ccc-cyfarwyddyd-saesneg \
  --seed 1 --workers 7
uv run --locked cymraeg-bench ccc-report runs/ccc-v0.1-beta.sqlite3
```

Nid yw'r dimensiynau CCC wedi'u cynnwys yn fformiwla sgôr gyffredinol v0.1.

### Trwydded

Mae cod CymraegBench ar gael o dan drwydded MIT. Gweler [LICENSE](LICENSE).

---

<a id="english"></a>

## English

An open, reproducible benchmark for evaluating Welsh-language models.
CymraegBench is the research harness behind the
[AIsteddfod](https://aisteddfod.com/) leaderboard; it brings together
BritEval, Techiaith and Cardiff NLP datasets and BangorAI protocols in one
transparent process.

It also includes the [CCC Language Audit](data/ccc-audit/README.md): 232 open
test cases that turn the qualitative findings of Kara Lewis's report on AI,
Welsh and higher education into seven reproducible subsets. The audit
measures correction and explanation separately, as well as idiomatic usage,
terminology, summarisation, quotations, references and the effect of the
instruction language.

[CymraegBench Voice v0.1](docs/VOICE.md) adds an ASR and TTS protocol: WER/CER,
latency and real-time factor for transcription, together with a CC0 prompt
pack and a blinded listening test for synthesis. The pinned catalogue
includes BangorAI Zipformer and Techiaith Whisper, Wav2Vec2, Kaldi and Piper
models.

See [PLAN.md](PLAN.md) for the protocol, dataset provenance and the rationale
behind the scoring choices. Smoke-test results are not publishable scores:
their purpose is to verify the harness, token limits and answer formats before
an official run.

### Getting started

```bash
uv sync --locked
cp .env.example .env
# add the three API keys to .env
uv run --locked --env-file .env cymraeg-bench validate
uv run --locked --env-file .env cymraeg-bench plan --max-cases 3
uv run --locked --env-file .env cymraeg-bench run --max-cases 3 --max-usd 5
uv run --locked --env-file .env cymraeg-bench report runs/<run-id>.sqlite3
uv run --locked --env-file .env cymraeg-bench ccc-report runs/<run-id>.sqlite3
```

`uv` manages Python 3.11, `.venv`, the dependencies and `uv.lock`. `--locked`
refuses to run if the lockfile does not match `pyproject.toml`; `--env-file
.env` loads the keys without storing them in the results database. A secrets
manager can be used instead of `.env`.

### Routing

- `OPENAI_API_KEY`: GPT-5.6 Sol through `POST /v1/responses`.
- `ANTHROPIC_API_KEY`: Claude Opus 4.8, Claude Opus 5 and Claude Fable 5
  through `POST /v1/messages`.
- `OPENROUTER_API_KEY`: Grok, Kimi, Muse Spark 1.2 (open weights), GLM and
  DeepSeek through `POST /api/v1/chat/completions`.
- Local models: start vLLM, TGI or Ollama with a Chat Completions-compatible
  interface and set the relevant `*_API_BASE`. Then set `enabled = true` in
  `config/models.toml`.

`validate --check-remote` compares OpenRouter IDs with the live catalogue
without sending a prompt to a model. `plan` makes no model calls and spends no
credit. `run` refuses to start if a required key is missing.

### Private data

Place the cynghanedd themes in:

```text
data/private/cynghanedd-themes.jsonl
{"id":"c001","theme":"..."}
```

Place licensed Barddas data in `data/private/barddas.jsonl`. Both paths are
ignored by Git. Set `CYNGHANEDD_SCORER_COMMAND` or `BARDDAS_SCORER_COMMAND` to
a program that reads JSON from stdin and returns JSON. The project does not
duplicate hidden or licensed content.

### Outputs

The public results include the
[original seven-model leaderboard](results/cymraegbench-v0.1/leaderboard.md)
and the
[extended 15-model leaderboard](results/cymraegbench-v0.1-estynedig/README.md).
The extended version includes full-dataset runs of Jupiter-N-120B and
Caernarfon 3B Cymraeg Instruct v0.1, as well as the earlier samples. The
[Claude Opus 5 run](results/cymraegbench-claude-opus-5-v0.1/leaderboard.md)
adds that model on the same deterministic sample, together with the CCC
Language Audit. The
[Muse Spark 1.2 run](results/cymraegbench-muse-spark-1-2-v0.1/leaderboard.md)
uses the same sample and scorers through OpenRouter.

Each run creates `runs/<run-id>.sqlite3`; `report` can generate Markdown and
CSV. The database retains the prompt, answer, reference, score, tokens, cost,
latency, actual model and provider metadata. It does not store API keys.

This Git repository does not redistribute third-party datasets. The adapters
download them from their original sources and pin the revision where
possible; each dataset's terms and licence continue to apply.

### Development

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked cymraeg-bench validate
uv run --locked cymraeg-bench plan --max-cases 1
```

After changing a dependency, use `uv add`, `uv remove` or `uv lock`, then
commit `pyproject.toml` and `uv.lock` together.

`aisteddfod-bench` is retained as a compatible alias for existing scripts.

For an official run, `--workers 7` can run up to one simultaneous call for
each model. Every SQLite write still takes place in the main process, and the
checkpoint retries only error rows when the same `--run-id` is resumed.

The CymraegBench v0.1 publication protocol uses up to 100 deterministic cases
from each suite, seed 1 and a safe cost limit:

```bash
uv run --locked --env-file .env cymraeg-bench run \
  --run-id cymraegbench-v0.1 \
  --max-cases 100 --seed 1 --workers 7 --max-usd 25
```

Once every call has completed without an error, generate the publishable
leaderboard and metadata as follows:

```bash
uv run --locked cymraeg-bench leaderboard runs/cymraegbench-v0.1.sqlite3
```

The overall score gives equal weight to the macro-average of nine Welsh
reasoning tests and the macro-average of nine practical Welsh tests. BLEU for
the legislation translation test is shown separately. If a call fails, only
that row is retried; `metadata.json` retains both the run's initial
configuration and the final configuration used for retries.

To run only the CCC audit:

```bash
uv run --locked --env-file .env cymraeg-bench run \
  --run-id ccc-v0.1-beta --suites ccc-cywiro-esbonio,ccc-priod-ddull,ccc-termau,ccc-crynhoi-dyfynnu,ccc-cyfeiriadau,ccc-cyfarwyddyd-cymraeg,ccc-cyfarwyddyd-saesneg \
  --seed 1 --workers 7
uv run --locked cymraeg-bench ccc-report runs/ccc-v0.1-beta.sqlite3
```

The CCC dimensions are not included in the v0.1 overall-score formula.

### Licence

CymraegBench is available under the MIT License. See [LICENSE](LICENSE).
