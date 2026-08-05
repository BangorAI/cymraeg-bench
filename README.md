# CymraegBench

Meincnod agored ac ailadroddadwy ar gyfer gwerthuso modelau iaith yn y
Gymraeg. CymraegBench yw'r harnais ymchwil sy'n sail i sgorfwrdd
[AIsteddfod](https://aisteddfod.com/); mae'n cyfuno setiau BritEval, Techiaith,
Cardiff NLP a phrotocolau BangorAI mewn un broses dryloyw.

Gweler [PLAN.md](PLAN.md) am y protocol, tarddiad y setiau a'r rhesymau dros y
dewisiadau sgorio. Nid yw canlyniadau rhediadau mwg yn sgoriau cyhoeddadwy:
eu diben yw gwirio'r harnais, terfynau tocynnau a fformat atebion cyn rhediad
swyddogol.

## Cychwyn

```bash
uv sync --locked
cp .env.example .env
# ychwanegu'r tair allwedd API at .env
uv run --locked --env-file .env cymraeg-bench validate
uv run --locked --env-file .env cymraeg-bench plan --max-cases 3
uv run --locked --env-file .env cymraeg-bench run --max-cases 3 --max-usd 5
uv run --locked --env-file .env cymraeg-bench report runs/<run-id>.sqlite3
```

Mae `uv` yn rheoli Python 3.11, `.venv`, y dibyniaethau a `uv.lock`. Mae
`--locked` yn gwrthod rhedeg os nad yw'r lockfile yn cyfateb i
`pyproject.toml`; mae `--env-file .env` yn llwytho'r allweddi heb eu cadw yn y
gronfa ganlyniadau. Gellir defnyddio rheolwr cyfrinachau yn lle `.env`.

## Llwybro

- `OPENAI_API_KEY`: GPT-5.6 Sol drwy `POST /v1/responses`.
- `ANTHROPIC_API_KEY`: Claude Opus 4.8 a Claude Fable 5 drwy
  `POST /v1/messages`.
- `OPENROUTER_API_KEY`: Grok, Kimi, GLM a DeepSeek drwy
  `POST /api/v1/chat/completions`.
- Modelau lleol: cychwyn vLLM/TGI/Ollama â rhyngwyneb Chat Completions a
  gosod yr `*_API_BASE` perthnasol. Wedyn troi `enabled = true` ymlaen yn
  `config/models.toml`.

Mae `validate --check-remote` yn cymharu IDs OpenRouter â'r catalog byw heb
anfon prompt at fodel. Nid yw `plan` yn gwneud unrhyw alwad fodel na gwario
credyd. Mae `run` yn gwrthod dechrau os oes allwedd ofynnol ar goll.

## Data preifat

Rhowch themâu'r gynghanedd yn:

```text
data/private/cynghanedd-themes.jsonl
{"id":"c001","theme":"..."}
```

Rhowch ddata trwyddedig Barddas yn `data/private/barddas.jsonl`. Mae'r ddau
lwybr wedi'u hanwybyddu gan Git. Gosodwch `CYNGHANEDD_SCORER_COMMAND` neu
`BARDDAS_SCORER_COMMAND` i raglen sy'n darllen JSON o stdin ac yn dychwelyd
JSON. Nid yw'r prosiect yn dyblygu cynnwys cudd na thrwyddedig.

## Allbynnau

Mae pob rhediad yn creu `runs/<run-id>.sqlite3`; gall `report` greu Markdown a
CSV. Mae'r gronfa'n cadw'r prompt, ateb, cyfeirnod, sgôr, tocynnau, cost,
latency, model gwirioneddol a metadata darparwr. Nid yw'n cadw allweddi API.

Nid yw'r storfa Git hon yn ailddosbarthu'r setiau trydydd parti. Mae'r
addaswyr yn eu lawrlwytho o'u ffynonellau gwreiddiol ac yn pinio'r fersiwn lle
bo modd; mae telerau a thrwydded pob set yn parhau'n gymwys.

## Datblygu

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

Mae'r prif sgôr yn rhoi pwysau cyfartal i gymedr macro naw prawf rhesymu
Cymraeg a chymedr macro naw prawf Cymraeg ymarferol. Dangosir BLEU y prawf
cyfieithu deddfwriaeth ar wahân. Os bydd galwad yn methu, ailgeisir y rhes
honno'n unig; mae `metadata.json` yn cadw cyfluniad dechrau'r rhediad a'r
cyfluniad terfynol a ddefnyddiwyd wrth ailgeisio.

## Trwydded

Mae cod CymraegBench ar gael o dan drwydded MIT. Gweler [LICENSE](LICENSE).
