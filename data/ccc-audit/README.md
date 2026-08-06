# Archwiliad Iaith CCC

Set agored o 232 o achosion i brofi'r mathau o fethiant a ddisgrifir yn
adroddiad Kara Lewis, *Deallusrwydd Artiffisial (AI), y Gymraeg ac Addysg
Uwch*. Nid atgynhyrchiad o ddeunydd prawf yr adroddiad yw hwn. Lluniwyd yr
enghreifftiau o'r newydd er mwyn rhoi prawf ailadroddadwy ar ei dacsonomeg.

## Cwmpas

| Ffeil | Achosion | Beth sy'n cael ei fesur |
|---|---:|---|
| `cywiro-esbonio.jsonl` | 64 | Cywiriad, dosbarth y gwall, y rheol a chywirdeb yr esboniad |
| `priod-ddull.jsonl` | 32 | Priod-ddull, cystrawen naturiol a Chymraeg Clir |
| `termau.jsonl` | 32 | Termau safonol, gan gynnwys bathiadau a nodwyd gan y CCC |
| `crynhoi-dyfynnu.jsonl` | 24 | Hepgoriadau, honiadau ychwanegol a chadw dyfyniad yn union |
| `cyfeiriadau.jsonl` | 32 | Ateb wedi'i seilio ar dystiolaeth, cyfeirio ac ymatal pan fo'r dystiolaeth ar goll |
| `cyfarwyddyd-cymraeg.jsonl` | 24 | Dewis ffynonellau dan gyfarwyddyd Cymraeg |
| `cyfarwyddyd-saesneg.jsonl` | 24 | Yr un prawf dan gyfarwyddyd Saesneg, gan gynnwys brawddeg Gymraeg y CCC |

Mae'r pâr olaf yn cynnwys tair amod ar gyfer pob pwnc: rheolaeth heb gyfyngu
iaith y ffynonellau; cyfarwyddyd i ddefnyddio ffynonellau Cymraeg yn unig; ac
amod sy'n defnyddio'r union frawddeg `Creu'r ymholiad gyda'r Gymraeg yn unig`.

## Sgorio

Mae'r sgorwyr yn benderfynedig ac nid ydynt yn defnyddio model arall fel
beirniad. Adroddir y dimensiynau ar wahân:

- cywiro ac esbonio: 50% cywiriad, 15% categori, 25% cod y rheol, 10% cynnwys
  yr esboniad;
- crynhoi: 30% dethol ffeithiau, 25% dyfyniad union, 40% cwmpas y crynodeb,
  5% peidio â dewis honiad anghywir;
- cyfeiriadau: 60% y cyfeiriadau cywir, 30% yr ateb, 10% peidio â dyfeisio ID;
- dewis ffynonellau: 80% yr union set, 20% peidio â dewis ffynhonnell
  annerbyniol.

Defnyddiwch `cymraeg-bench ccc-report` i gael tabl fesul dimensiwn. Ni chaiff
y sgoriau eu cymysgu â sgôr gyffredinol CymraegBench v0.1.

## Sail ieithyddol

Defnyddiwyd y ffynonellau hyn wrth lunio ac adolygu'r achosion:

- Kara Lewis, *Deallusrwydd Artiffisial (AI), y Gymraeg ac Addysg Uwch*,
  PCYDDS / Y Coleg Cymraeg Cenedlaethol, 2026: tacsonomeg y methiannau;
- [Yr Arddulliadur, BydTermCymru](https://www.gov.wales/bydtermcymru/style-guide):
  `a/ac`, `ni/nid`, treigladau, berfau cynorthwyol, Cymraeg Clir, `mwy/rhagor`
  a thermau safonol;
- [Canllawiau Iaith BBC Cymru Fyw](https://downloads.bbc.co.uk/cymru/gwybodaeth/CanllawiauIaithBBCCymruFyw.pdf):
  ail gyfeirbwynt ar arddull Gymraeg;
- [Nodiadau'r Treigladau, Hwb](https://resources.hwb.gov.wales/VTC/ngfl/welsh/127/Nodiadau/11_Treigladau.pdf):
  patrymau treiglo;
- geirfa gyhoeddedig AIsteddfod: termau AI y prosiect.

Mae catalogau'r profion crynhoi, cyfeirio a chwilio yn ffuglennol ac wedi'u
labelu'n glir yn y data. Prawf byd caeedig yw hwn: nid yw'n honni bod y
ffynonellau synthetig yn gyhoeddiadau go iawn.

## Ansawdd a fersiynu

Mae `manifest.json` yn cofnodi nifer yr achosion a SHA-256 pob ffeil. Mae
`scripts/build_ccc_audit_data.py` yn ailgynhyrchu'r ffeiliau'n union. Rhaid
newid rhif y fersiwn a chofnodi unrhyw newid i ateb cywir ar ôl i ganlyniadau
swyddogol gael eu cyhoeddi.

Statws `0.1.0-beta` sydd i'r set nes bod panel annibynnol o o leiaf ddau
olygydd Cymraeg wedi adolygu pob achos. Dylid adrodd sgoriau beta fel
canlyniadau ymchwil, nid fel dyfarniad terfynol am ansawdd model.

## Trwydded

Gweler [LICENSE.md](LICENSE.md). Mae'r data ar gael dan Drwydded Llywodraeth
Agored v3.0, gyda'r priodoliadau a nodir yno.
