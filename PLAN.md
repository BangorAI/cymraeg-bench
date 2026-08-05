# Cynllun gwerthuso CymraegBench

## Nod

Mesur gallu modelau iaith i ddeall a chynhyrchu Cymraeg drwy un harnais
ailadroddadwy. Defnyddir OpenRouter ar gyfer y modelau trydydd parti sydd ar
gael yno, ac APIs OpenAI ac Anthropic yn uniongyrchol er mwyn defnyddio'r
allweddi ar wahân sydd eisoes gan y prosiect. Caiff modelau pwysau agored
Cymraeg eu gwasanaethu drwy endpoint lleol sy'n gydnaws ag OpenAI.

## Matrics y modelau

| Llwybr | Modelau | Statws diofyn |
|---|---|---|
| OpenAI uniongyrchol | GPT-5.6 Sol | ymlaen |
| Anthropic uniongyrchol | Claude Opus 4.8; Claude Fable 5 | ymlaen |
| OpenRouter | Grok 4.20; Kimi K3; GLM-5.2; DeepSeek V4-Flash-0731 | ymlaen |
| Lleol / OpenAI-compatible | Jupiter-N-120B; Caernarfon 3B; Mwydryn | wedi'u diffodd nes bod endpoint ar gael |
| Cyfeirnodau lleol | Mistral 7B; Llama 3 8B; BangorAI Mistral Cymraeg v2 | dewisol |

Mae `config/models.toml` yn pinio union ID pob model. Dylid rhedeg `validate`
cyn pob ymgyrch i ganfod model sydd wedi diflannu neu allwedd sydd ar goll.

## Setiau profion

1. BritEval: ARC-Easy, ARC-Challenge, PIQA, TruthfulQA-MC1 ac XNLI-Cymraeg.
2. Techiaith `llm-evals-cy`: y deg prawf gwreiddiol, wedi'u pinio i commit
   `47839d2147c97fd2f10a52dd36751608e5fa36bf`.
3. Techiaith ychwanegol: `mgsm_cy`, `COPA-cy` a `wnli-cy`.
4. Cardiff NLP: `welsh-cefr` fel tasg ddosbarthu sero-ergyd.
5. Bangor AI: protocol cudd y gynghanedd (40 thema x 5 cynnig), drwy ddilysydd
   lleol; a Barddas fel meincnod dilysydd ar wahân, nid prawf sgwrsio cyffredinol.

Mae hyn yn rhoi 19 cyfluniad cyhoeddus (mae ARC-Easy ac ARC-Challenge yn cael
eu cyfrif ar wahân), ynghyd â'r ddau lwybr preifat.

Ni chaiff y themâu cudd na data Barddas eu cynnwys yn y storfa. Rhaid eu rhoi
yn `data/private/` a darparu'r gorchymyn dilysu drwy newidyn amgylchedd.

## Protocol teg

- Defnyddio'r un negeseuon system a defnyddiwr ar draws pob darparwr.
- `temperature=0` a `seed=1` lle mae'r darparwr yn eu cynnal; cofnodi pan fydd
  paramedr yn cael ei hepgor.
- Dim offer, chwilio gwe na dolen adborth. Ateb byr yn unig ar dasgau
  amlddewis; dim cadwyn feddwl yn y canlyniad.
- Pinio fersiwn y set, ID y model, ymateb crai, defnydd tocynnau, cost, amser,
  statws gwrthod a metadata llwybro OpenRouter.
- Defnyddio set prawf/validation swyddogol; samplu penderfynedig gyda'r un
  `seed` os defnyddir `--max-cases`.
- Sgorio ARC/PIQA/COPA/MMLU/XNLI/WNLI/CEFR drwy union gyfatebiaeth wedi'i
  normaleiddio; MGSM drwy ateb rhifol; cyfieithu drwy corpus BLEU; TruthfulQA
  drwy MC1. Nid yw MC2 tebygolrwydd yn deg heb log-probabilities cyson.

## Camau rhedeg

1. `uv sync --locked`, ac wedyn `uv run --locked --env-file .env
   cymraeg-bench validate --check-remote`, i wirio'r amgylchedd, y modelau
   a'r ffynonellau.
2. Rhediad mwg: pob model x pob set drwy `uv run`, gyda `--max-cases 3`.
3. Adolygu gwallau, gwrthodiadau, cost a fformat atebion.
4. Rhediad llawn wrth grŵp (objective, generation, private) gyda `--max-usd`
   a checkpoint SQLite; ailgychwyn yr un `--run-id` os oes toriad.
5. Cynhyrchu adroddiad Markdown/CSV, yna archwilio sampl o atebion â llaw cyn
   cyhoeddi unrhyw sgorfwrdd.

## Sicrhau ansawdd

Mae'r profion uned yn ffugio pob API; ni wariant arian. Nid yw allweddi byth yn
cael eu hargraffu na'u cadw. Mae pob rhes SQLite yn unigryw yn ôl rhediad,
model, set, achos ac ailadroddiad, felly mae ailddechrau'n ddiogel.
Mae `uv.lock` a `.python-version` yn pinio'r amgylchedd gweithredu, a dylid
defnyddio `uv run --locked` ar gyfer rhediadau swyddogol.
