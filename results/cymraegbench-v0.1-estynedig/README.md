# CymraegBench v0.1: y sgorfwrdd estynedig

Mae'r sgorfwrdd hwn yn cyfuno 13 model a 88,273 o ymatebion. Cafodd
Jupiter-N-120B a Chaernarfon 3B Cymraeg Instruct v0.1 eu rhedeg dros y set
gyhoeddus lawn: 33,657 o achosion yr un. Mae'r 11 model blaenorol yn samplau
hyd at 100 achos o bob set. Defnyddir yr un had, sgorwyr a fformiwla ym mhob
rhediad, ond dylid cofio bod llai o ansicrwydd samplu yn y ddau rediad llawn.

Mae Jupiter yn sgorio 48.87 ac yn dod yn wythfed. Mae Caernarfon Cymraeg
Instruct yn sgorio 10.83 ac yn dod yn ddeuddegfed. Dim ond y 19 set graidd
sy'n cyfrannu at sgôr gyffredinol CymraegBench; adroddir saith dimensiwn
Archwiliad Iaith CCC ar wahân.

## Protocol

- tymheredd 0 a had 1; hyd at 100 achos o bob set ar gyfer y samplau, a phob
  achos ar gyfer y ddau rediad set lawn;
- allbwn gwag neu allbwn sy'n taro'r terfyn tocynnau: `invalid`, sgôr sero;
- Jupiter-N-120B: pwysau BF16, gwasanaeth vLLM a lleiafswm o 512 tocyn
  allbwn wrth ailbrofi;
- Caernarfon Cymraeg Instruct: addasydd LoRA dros britllm-3b-v0.1 a therfyn
  cyd-destun brodorol o 2,048;
- Mwydryn Phi-2: fformat Alpaca;
- Mistral Cymraeg v2 a Mwydryn 7B v2: fformat Llama 2 a nodir ar gardiau'r
  modelau;
- Techiaith Llama 3.2 1B SFT: union ragymadrodd Alpaca demo swyddogol
  Techiaith a `repetition_penalty = 1.15`.

Mae rhediad y model sylfaenol Caernarfon 3B yn dal yn yr archif er mwyn
tryloywder, ond mae wedi'i dynnu o'r prif sgorfwrdd gan fod y fersiwn instruct
bellach yn fesur mwy defnyddiol. Cofnodwyd cyfyngiadau cyd-destun y model fel
allbynnau annilys â sgôr sero; nid ydynt yn wallau seilwaith.

## Rhediadau ffynhonnell

- `cymraegbench-v0.1`: y saith model blaenllaw;
- `cymraegbench-v0.1-lleol-bangor-mistral-cymraeg-v2`;
- `cymraegbench-v0.1-lleol-mwydryn`;
- `cymraegbench-v0.1-lleol-mwydryn-7b-v2`;
- `cymraegbench-v0.1-lleol-techiaith-llama-3-2-1b-sft`;
- [`jupiter-spark-bf16-v0.1`](../jupiter-spark-bf16-v0.1/README.md);
- [`caernarfon-3b-cymraeg-instruct-v0.1`](../caernarfon-3b-cymraeg-instruct-v0.1/README.md).

Gweler [leaderboard.md](leaderboard.md) am y safleoedd,
[suite-scores.md](suite-scores.md) am bob set, a [metadata.json](metadata.json)
am gyfluniad y modelau a chyfrifiad y sgoriau. Mae
[y llyfr gwaith](cymraegbench-v0.1-estynedig.xlsx) yn darparu'r un data mewn
fformat taenlen.
