# CymraegBench v0.1: y sgorfwrdd estynedig

Mae'r sgorfwrdd hwn yn cyfuno'r rhediad saith-model gwreiddiol â phum model
agored a redwyd yn lleol ar un RTX 4090. Ceir 12 model a 23,012 o ymatebion
yn y gronfa gyfun.

Nid un swp API oedd hwn. Cafodd y saith model blaenllaw 1,821 o achosion yr
un ar y 19 set graidd. Cafodd y pum model lleol 2,053 o achosion yr un: yr un
19 set graidd, ynghyd â 232 o achosion Archwiliad Iaith CCC. Dim ond y 19 set
graidd sy'n cyfrannu at sgôr gyffredinol CymraegBench; adroddir y dimensiynau
CCC ar wahân.

## Protocol y modelau lleol

- tymheredd 0, had 1, a hyd at 100 achos o bob set;
- allbwn gwag neu allbwn sy'n taro'r terfyn tocynnau: `invalid`, sgôr sero;
- Caernarfon: prompt testun plaen, gan nad model cyfarwyddyd mohono;
- Mwydryn Phi-2: fformat Alpaca;
- Mistral Cymraeg v2 a Mwydryn 7B v2: fformat Llama 2 a nodir ar gardiau'r
  modelau;
- Techiaith Llama 3.2 1B SFT: union ragymadrodd Alpaca demo swyddogol
  Techiaith a `repetition_penalty = 1.15`.

Mae Caernarfon 3B yn llinell sylfaen rag-hyfforddedig, nid model sydd wedi'i
fireinio i ddilyn cyfarwyddiadau. Dylid darllen ei sgôr sero a'i gyfradd uchel
o allbynnau annilys yn y cyd-destun hwnnw; nid yw'n honiad nad oes gwybodaeth
Gymraeg yn ei bwysau.

## Rhediadau ffynhonnell

- `cymraegbench-v0.1`: y saith model blaenllaw;
- `cymraegbench-v0.1-lleol-bangor-mistral-cymraeg-v2`;
- `cymraegbench-v0.1-lleol-mwydryn`;
- `cymraegbench-v0.1-lleol-caernarfon-3b`;
- `cymraegbench-v0.1-lleol-mwydryn-7b-v2`;
- `cymraegbench-v0.1-lleol-techiaith-llama-3-2-1b-sft`.

Gweler [leaderboard.md](leaderboard.md) am y safleoedd,
[suite-scores.md](suite-scores.md) am bob set, a [metadata.json](metadata.json)
am gyfluniad y modelau a chyfrifiad y sgoriau.
