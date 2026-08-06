from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ccc-audit"

EDIT_SYSTEM = """Rwyt ti'n brawfddarllen Cymraeg ysgrifenedig safonol. Mae union un gwall neu wall priod-ddull yn y frawddeg. Dychwela wrthrych JSON yn unig â'r pedwar maes hyn: {\"cywiriad\": \"y frawddeg gyfan\", \"categori\": \"label\", \"rheol\": \"COD\", \"esboniad\": \"esboniad byr Cymraeg\"}.

Labeli categori: treiglad, cysylltair, arddodiad, negyddiaeth, cystrawen, term.
Codau rheol: TM_YN_TRAETHIADOL, A_AC_LLAFARIAD, A_AC_CYTSAIN, YN_ENW_PENDANT, MEWN_ENW_AMPHENODOL, NEG_BOD_NID, NEG_LLAES, NEG_MEDDAL, NEG_NI, CYST_AIL_LUNIO, TERM_SAFONOL, PWYSLAIS_DIANGEN."""

CHOICE_SYSTEM = (
    "Dewisa'r unig ateb gorau ar gyfer Cymraeg ysgrifenedig safonol. "
    "Ateba ag A neu B yn unig, heb esboniad nac atalnodi."
)


def row(case_id: str, system: str, user: str, ideal: object, **metadata: object) -> dict[str, object]:
    return {
        "id": case_id,
        "system": system,
        "user": user,
        "ideal": ideal,
        "metadata": metadata,
    }


def edit_row(
    case_id: str,
    wrong: str,
    correct: str,
    category: str,
    rule: str,
    concepts: list[list[str]],
    basis: str,
) -> dict[str, object]:
    return row(
        case_id,
        EDIT_SYSTEM,
        f"Cywira ac esbonia'r frawddeg hon:\n\n{wrong}",
        {
            "accepted_corrections": [correct],
            "category": category,
            "rule": rule,
            "explanation_concepts": concepts,
        },
        dimension="cywiro_esbonio",
        phenomenon=rule,
        basis=basis,
    )


def build_editing() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    # The first member is deliberately wrong; keeping the pairs explicit makes review straightforward.
    predicative = [
        ("Mae'r adroddiad yn pwysig.", "Mae'r adroddiad yn bwysig."),
        ("Mae'r esboniad yn cywir.", "Mae'r esboniad yn gywir."),
        ("Mae'r rhaglen yn defnyddiol.", "Mae'r rhaglen yn ddefnyddiol."),
        ("Mae'r dystiolaeth yn cadarn.", "Mae'r dystiolaeth yn gadarn."),
        ("Mae'r canlyniad yn teg.", "Mae'r canlyniad yn deg."),
        ("Mae'r dull yn priodol.", "Mae'r dull yn briodol."),
        ("Mae'r crynodeb yn cryno.", "Mae'r crynodeb yn gryno."),
        ("Mae'r ymateb yn manwl.", "Mae'r ymateb yn fanwl."),
        ("Mae'r system yn diogel.", "Mae'r system yn ddiogel."),
        ("Mae'r frawddeg yn byr.", "Mae'r frawddeg yn fyr."),
        ("Mae'r cynnig yn beiddgar.", "Mae'r cynnig yn feiddgar."),
        ("Mae'r ateb yn boddhaol.", "Mae'r ateb yn foddhaol."),
    ]
    for index, (wrong, correct) in enumerate(predicative, 1):
        cases.append(
            edit_row(
                f"tm-{index:02d}", wrong, correct, "treiglad", "TM_YN_TRAETHIADOL",
                [["treiglad meddal"], ["yn traethiadol", "yn o flaen ansoddair", "yn + ansoddair"]],
                "CCC: treigladau; BydTermCymru: yn traethiadol",
            )
        )

    conjunctions = [
        ("Mae'r prosiect yn cyfuno data a ymchwil.", "Mae'r prosiect yn cyfuno data ac ymchwil.", "A_AC_LLAFARIAD"),
        ("Mae'r cwrs yn trafod iaith a addysg.", "Mae'r cwrs yn trafod iaith ac addysg.", "A_AC_LLAFARIAD"),
        ("Prynwyd offer a adnoddau newydd.", "Prynwyd offer ac adnoddau newydd.", "A_AC_LLAFARIAD"),
        ("Cymharwyd Cymru a Ewrop.", "Cymharwyd Cymru ac Ewrop.", "A_AC_LLAFARIAD"),
        ("Mae angen gwaith a amser.", "Mae angen gwaith ac amser.", "A_AC_LLAFARIAD"),
        ("Defnyddiwyd modelau a algorithmau gwahanol.", "Defnyddiwyd modelau ac algorithmau gwahanol.", "A_AC_LLAFARIAD"),
        ("Cofnodwyd llais ac testun.", "Cofnodwyd llais a thestun.", "A_AC_CYTSAIN"),
        ("Dadansoddwyd delweddau ac modelau.", "Dadansoddwyd delweddau a modelau.", "A_AC_CYTSAIN"),
        ("Casglwyd sain ac data.", "Casglwyd sain a data.", "A_AC_CYTSAIN"),
        ("Cyhoeddwyd ffigurau ac lluniau.", "Cyhoeddwyd ffigurau a lluniau.", "A_AC_CYTSAIN"),
        ("Adolygwyd brawddegau ac geiriau.", "Adolygwyd brawddegau a geiriau.", "A_AC_CYTSAIN"),
        ("Gofynnwyd am dystiolaeth ac barn.", "Gofynnwyd am dystiolaeth a barn.", "A_AC_CYTSAIN"),
    ]
    for index, (wrong, correct, rule) in enumerate(conjunctions, 1):
        concepts = (
            [["ac"], ["o flaen llafariad", "cyn llafariad"]]
            if rule.endswith("LLAFARIAD")
            else [["a"], ["o flaen cytsain", "cyn cytsain"]]
        )
        cases.append(
            edit_row(
                f"ac-{index:02d}", wrong, correct, "cysylltair", rule, concepts,
                "CCC: a/ac; BydTermCymru: a / ac",
            )
        )

    prepositions = [
        ("Mae'r disgyblion mewn yr ysgol.", "Mae'r disgyblion yn yr ysgol.", "YN_ENW_PENDANT"),
        ("Mae'r tîm mewn y swyddfa.", "Mae'r tîm yn y swyddfa.", "YN_ENW_PENDANT"),
        ("Cynhaliwyd y prawf mewn y labordy.", "Cynhaliwyd y prawf yn y labordy.", "YN_ENW_PENDANT"),
        ("Ceir y manylion mewn yr adroddiad.", "Ceir y manylion yn yr adroddiad.", "YN_ENW_PENDANT"),
        ("Mae'r enghraifft mewn y llyfr.", "Mae'r enghraifft yn y llyfr.", "YN_ENW_PENDANT"),
        ("Cedwir y cofnodion mewn y gronfa ddata.", "Cedwir y cofnodion yn y gronfa ddata.", "YN_ENW_PENDANT"),
        ("Cyhoeddwyd yr erthygl yn cylchgrawn academaidd.", "Cyhoeddwyd yr erthygl mewn cylchgrawn academaidd.", "MEWN_ENW_AMPHENODOL"),
        ("Gosodwyd y data yn cronfa agored.", "Gosodwyd y data mewn cronfa agored.", "MEWN_ENW_AMPHENODOL"),
        ("Bu'r tîm yn gweithio yn swyddfa fach.", "Bu'r tîm yn gweithio mewn swyddfa fach.", "MEWN_ENW_AMPHENODOL"),
        ("Cynhaliwyd y prawf yn labordy newydd.", "Cynhaliwyd y prawf mewn labordy newydd.", "MEWN_ENW_AMPHENODOL"),
        ("Trafodwyd y mater yn cyfarfod cyhoeddus.", "Trafodwyd y mater mewn cyfarfod cyhoeddus.", "MEWN_ENW_AMPHENODOL"),
        ("Cafodd y canlyniadau eu cynnwys yn adroddiad blynyddol.", "Cafodd y canlyniadau eu cynnwys mewn adroddiad blynyddol.", "MEWN_ENW_AMPHENODOL"),
    ]
    for index, (wrong, correct, rule) in enumerate(prepositions, 1):
        concepts = (
            [["yn"], ["enw pendant", "y fannod", "y neu yr"]]
            if rule.startswith("YN_")
            else [["mewn"], ["enw amhenodol", "heb y fannod"]]
        )
        cases.append(
            edit_row(
                f"ym-{index:02d}", wrong, correct, "arddodiad", rule, concepts,
                "CCC: mewn/yn",
            )
        )

    negatives = [
        ("Nid mae'r canlyniad yn derfynol.", "Nid yw'r canlyniad yn derfynol.", "NEG_BOD_NID", [["nid yw"], ["ffurf negyddol", "negydd"]]),
        ("Ni oedd y data'n gyflawn.", "Nid oedd y data'n gyflawn.", "NEG_BOD_NID", [["nid oedd"], ["berf bod", "bod"]]),
        ("Ni oes tystiolaeth ddigonol.", "Nid oes tystiolaeth ddigonol.", "NEG_BOD_NID", [["nid oes"], ["berf bod", "bod"]]),
        ("Ni cafodd y cais ei dderbyn.", "Ni chafodd y cais ei dderbyn.", "NEG_LLAES", [["treiglad llaes"], ["c yn troi'n ch", "c i ch"]]),
        ("Ni parhaodd y prawf yn hir.", "Ni pharhaodd y prawf yn hir.", "NEG_LLAES", [["treiglad llaes"], ["p yn troi'n ph", "p i ph"]]),
        ("Ni talodd y sefydliad y ffi.", "Ni thalodd y sefydliad y ffi.", "NEG_LLAES", [["treiglad llaes"], ["t yn troi'n th", "t i th"]]),
        ("Nid welodd y tîm y gwall.", "Ni welodd y tîm y gwall.", "NEG_NI", [["ni"], ["o flaen cytsain", "cyn cytsain"]]),
        ("Nid ddaeth yr ateb mewn pryd.", "Ni ddaeth yr ateb mewn pryd.", "NEG_NI", [["ni"], ["o flaen cytsain", "cyn cytsain"]]),
        ("Ni bydd y newid yn effeithio ar y data.", "Ni fydd y newid yn effeithio ar y data.", "NEG_MEDDAL", [["treiglad meddal"], ["b yn troi'n f", "b i f"]]),
        ("Nid roddodd y model reswm.", "Ni roddodd y model reswm.", "NEG_NI", [["ni"], ["o flaen cytsain", "cyn cytsain"]]),
        ("Nid ellir cadarnhau'r honiad.", "Ni ellir cadarnhau'r honiad.", "NEG_NI", [["ni ellir"], ["ffurf negyddol", "negydd"]]),
        ("Ni dylid newid y dyfyniad.", "Ni ddylid newid y dyfyniad.", "NEG_MEDDAL", [["treiglad meddal"], ["d yn troi'n dd", "d i dd"]]),
    ]
    for index, (wrong, correct, rule, concepts) in enumerate(negatives, 1):
        cases.append(
            edit_row(
                f"neg-{index:02d}", wrong, correct, "negyddiaeth", rule, concepts,
                "CCC: ffurfiau negyddol; BydTermCymru: y negydd berfol",
            )
        )

    calques = [
        ("Mae'r tîm yn gwneud defnydd o'r data.", "Mae'r tîm yn defnyddio'r data.", [["defnyddio"], ["berf gynorthwyol", "cystrawen Saesneg", "cryno"]]),
        ("Talodd y panel ymweliad â'r ganolfan.", "Ymwelodd y panel â'r ganolfan.", [["ymweld"], ["berf gynorthwyol", "cystrawen Saesneg", "cryno"]]),
        ("Mae hyn yn cynrychioli cam ymlaen.", "Mae hyn yn gam ymlaen.", [["yn gam ymlaen"], ["cystrawen Saesneg", "berf ddiangen"]]),
        ("Mae hyn yn gwasanaethu fel rhybudd.", "Mae hyn yn rhybudd.", [["yn rhybudd"], ["cystrawen Saesneg", "berf ddiangen"]]),
        ("Bydd y pwyllgor yn rhoi ystyriaeth i'r dystiolaeth.", "Bydd y pwyllgor yn ystyried y dystiolaeth.", [["ystyried"], ["cystrawen Saesneg", "berf syml"]]),
        ("Bydd y panel yn cymryd ystyriaeth o'r effaith.", "Bydd y panel yn ystyried yr effaith.", [["ystyried"], ["cystrawen Saesneg", "berf syml"]]),
        ("Gwnaeth y tîm ymgais i ddatrys y broblem.", "Ceisiodd y tîm ddatrys y broblem.", [["ceisiodd"], ["cystrawen Saesneg", "berf syml"]]),
        ("Gwnaeth y bwrdd benderfyniad i ohirio'r lansiad.", "Penderfynodd y bwrdd ohirio'r lansiad.", [["penderfynodd"], ["cystrawen Saesneg", "berf syml"]]),
    ]
    for index, (wrong, correct, concepts) in enumerate(calques, 1):
        cases.append(
            edit_row(
                f"cyst-{index:02d}", wrong, correct, "cystrawen", "CYST_AIL_LUNIO", concepts,
                "CCC: cystrawen Seisnig; BydTermCymru: berfau cynorthwyol",
            )
        )

    terms = [
        ("\"Apps\" yw benthyciad o'r Saesneg.", "Mae \"apps\" yn fenthyciad o'r Saesneg.", "cystrawen", "PWYSLAIS_DIANGEN", [["pwyslais"], ["mae", "cystrawen arferol"]]),
        ("Mae'r dull yn hunanganolig.", "Mae'r dull yn hunanganolog.", "term", "TERM_SAFONOL", [["hunanganolog"], ["term safonol", "ffurf safonol"]]),
        ("Mae'r fframwaith yn theorïol.", "Mae'r fframwaith yn ddamcaniaethol.", "term", "TERM_SAFONOL", [["damcaniaethol"], ["term safonol", "ffurf safonol"]]),
        ("Mae'r ymchwil yn trafod ymddygiadoliaeth.", "Mae'r ymchwil yn trafod ymddygiadaeth.", "term", "TERM_SAFONOL", [["ymddygiadaeth"], ["term safonol", "ffurf safonol"]]),
        ("Lawrlwythwyd tri apps i'r ddyfais.", "Lawrlwythwyd tri ap i'r ddyfais.", "term", "TERM_SAFONOL", [["ap"], ["ffurf Gymraeg", "term Cymraeg"]]),
        ("Mae'r model yn cynnig esboniad theorïol.", "Mae'r model yn cynnig esboniad damcaniaethol.", "term", "TERM_SAFONOL", [["damcaniaethol"], ["term safonol", "ffurf safonol"]]),
        ("Dull hunanganolig oedd hwn.", "Dull hunanganolog oedd hwn.", "term", "TERM_SAFONOL", [["hunanganolog"], ["term safonol", "ffurf safonol"]]),
        ("Cymharwyd ymddygiadoliaeth â gwybyddiaeth.", "Cymharwyd ymddygiadaeth â gwybyddiaeth.", "term", "TERM_SAFONOL", [["ymddygiadaeth"], ["term safonol", "ffurf safonol"]]),
    ]
    for index, (wrong, correct, category, rule, concepts) in enumerate(terms, 1):
        cases.append(
            edit_row(
                f"term-{index:02d}", wrong, correct, category, rule, concepts,
                "CCC: gwallau yn esboniadau'r modelau",
            )
        )
    assert len(cases) == 64
    return cases


def choice_rows(prefix: str, pairs: list[tuple[str, str, str]], question: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (preferred, rejected, category) in enumerate(pairs, 1):
        if index % 2:
            options, answer = (preferred, rejected), "A"
        else:
            options, answer = (rejected, preferred), "B"
        rows.append(
            row(
                f"{prefix}-{index:02d}", CHOICE_SYSTEM,
                f"{question}\n\nA: {options[0]}\nB: {options[1]}", answer,
                dimension=prefix, category=category,
            )
        )
    return rows


def build_idiom() -> list[dict[str, object]]:
    pairs = [
        ("Mae'r tîm yn defnyddio'r data.", "Mae'r tîm yn gwneud defnydd o'r data.", "berfau_cynorthwyol"),
        ("Ymwelodd y panel â'r ganolfan.", "Talodd y panel ymweliad â'r ganolfan.", "berfau_cynorthwyol"),
        ("Mae hyn yn gam ymlaen.", "Mae hyn yn cynrychioli cam ymlaen.", "berfau_cynorthwyol"),
        ("Mae hyn yn rhybudd.", "Mae hyn yn gwasanaethu fel rhybudd.", "berfau_cynorthwyol"),
        ("Bydd y pwyllgor yn ystyried y dystiolaeth.", "Bydd y pwyllgor yn rhoi ystyriaeth i'r dystiolaeth.", "berfau_cynorthwyol"),
        ("Bydd y panel yn ystyried yr effaith.", "Bydd y panel yn cymryd ystyriaeth o'r effaith.", "berfau_cynorthwyol"),
        ("Ceisiodd y tîm ddatrys y broblem.", "Gwnaeth y tîm ymgais i ddatrys y broblem.", "berfau_cynorthwyol"),
        ("Ymdrechodd y gwasanaeth i wella.", "Gwnaeth y gwasanaeth ymdrech i wella.", "berfau_cynorthwyol"),
        ("Penderfynodd y bwrdd ohirio.", "Gwnaeth y bwrdd benderfyniad i ohirio.", "berfau_cynorthwyol"),
        ("Daliwch ati â'r gwaith.", "Cariwch ymlaen efo'r gwaith.", "berfau_ymadroddol"),
        ("Bydd y tîm yn ymchwilio i'r mater.", "Bydd y tîm yn edrych i mewn i'r mater.", "berfau_ymadroddol"),
        ("Gwrthododd y panel y cynnig.", "Trodd y panel y cynnig i lawr.", "berfau_ymadroddol"),
        ("Gohiriwyd y cyfarfod.", "Rhoddwyd y cyfarfod i ffwrdd.", "berfau_ymadroddol"),
        ("Cyrhaeddodd y siaradwr yn hwyr.", "Trodd y siaradwr i fyny'n hwyr.", "berfau_ymadroddol"),
        ("Mae'r cynllun yn bodloni'r gofynion.", "Mae'r cynllun yn cwrdd â'r gofynion.", "meet"),
        ("Bydd y grant yn talu'r costau.", "Bydd y grant yn cwrdd â'r costau.", "meet"),
        ("Cwrddodd y myfyrwyr â'r tiwtor.", "Bodlonodd y myfyrwyr y tiwtor.", "meet"),
        ("Cysylltwch â ni i gael rhagor o wybodaeth.", "Cysylltwch â ni i gael mwy o wybodaeth.", "mwy_rhagor"),
        ("Cafwyd mwy o geisiadau nag y llynedd.", "Cafwyd rhagor o geisiadau nag y llynedd.", "mwy_rhagor"),
        ("Cyflwynodd y bwrdd ganllawiau newydd.", "Cafodd canllawiau newydd eu cyflwyno gan y bwrdd.", "llais_gweithredol"),
        ("Dadansoddodd yr ymchwilwyr y data.", "Cafodd y data eu dadansoddi gan yr ymchwilwyr.", "llais_gweithredol"),
        ("Tra oeddem yn teithio, cofnodwyd y sain.", "Tra'n teithio, cofnodwyd y sain.", "tra"),
        ("Mae'r papur hwn yn egluro'r dull.", "Mae'r papur yma yn egluro'r dull.", "cywair_ffurfiol"),
        ("Mae'r rheoliadau hyn yn gymwys.", "Mae'r rheoliadau yma yn gymwys.", "cywair_ffurfiol"),
        ("Mae'r adroddiad yn disgrifio'r canlyniadau.", "Disgrifir y canlyniadau gan yr adroddiad.", "llais_gweithredol"),
        ("Bydd y cynllun yn helpu busnesau bach.", "Bydd cymorth yn cael ei ddarparu i fusnesau bach gan y cynllun.", "llais_gweithredol"),
        ("Byddwn yn gwerthuso'r gwasanaeth.", "Byddwn yn cynnal gwerthusiad o'r gwasanaeth.", "berfenw"),
        ("Mae'r tîm yn datblygu'r adnodd.", "Mae'r tîm yn gwneud gwaith datblygu ar yr adnodd.", "berfenw"),
        ("Rhaid inni leihau'r gost.", "Mae'n angenrheidiol ein bod yn gwneud gostyngiad yn y gost.", "cymraeg_clir"),
        ("Esboniwch y newid yn gryno.", "Rhowch esboniad cryno ynghylch y newid.", "cymraeg_clir"),
        ("Mae'r ap yn dangos y canlyniad.", "Mae'r canlyniad yn cael ei ddangos gan yr ap.", "llais_gweithredol"),
        ("Bydd y pwyllgor yn adolygu'r polisi.", "Bydd adolygiad o'r polisi yn cael ei wneud gan y pwyllgor.", "llais_gweithredol"),
    ]
    return choice_rows(
        "priod", pairs,
        "Pa frawddeg sy'n fwy naturiol a phriodol mewn testun ffurfiol?",
    )


def build_terms() -> list[dict[str, object]]:
    pairs = [
        ("deallusrwydd artiffisial", "deallusrwydd artificial", "AI"),
        ("AI cynhyrchiol", "AI cynhyrchiadol", "AI"),
        ("Model Iaith Mawr", "Model Iaith Fawr", "AI"),
        ("data hyfforddi", "data hyfforddiant", "AI"),
        ("cyfieithu peirianyddol", "cyfieithu peiriant", "AI"),
        ("lleferydd-i-destun", "llais i destun", "AI"),
        ("testun-i-leferydd", "testun i lais", "AI"),
        ("sgwrsfot", "chatbot", "AI"),
        ("rhyngwyneb defnyddiwr", "interface defnyddiwr", "AI"),
        ("rhithwelediad", "hallucination", "AI"),
        ("uniondeb academaidd", "integriti academaidd", "addysg"),
        ("camymddwyn academaidd", "misconduct academaidd", "addysg"),
        ("llythrennedd AI", "literasi AI", "addysg"),
        ("cyfrifoldeb digidol", "cyfrifoldeb digital", "addysg"),
        ("mireinio model", "fine-tunio model", "AI"),
        ("pwysau agored", "pwysau open-source", "AI"),
        ("cerdyn model", "card model", "AI"),
        ("meincnod", "benchmark", "AI"),
        ("rhoi cyfarwyddiadau", "promptio", "AI"),
        ("corpws", "corpus", "ieithyddiaeth"),
        ("apiau", "apps", "benthyg"),
        ("damcaniaethol", "theorïol", "CCC"),
        ("ymddygiadaeth", "ymddygiadoliaeth", "CCC"),
        ("hunanganolog", "hunanganolig", "CCC"),
        ("posibl", "posib", "safon_ty"),
        ("trafnidiaeth gyhoeddus", "cludiant cyhoeddus", "trafnidiaeth"),
        ("nifer o geisiadau", "rhif o geisiadau", "rhif_nifer"),
        ("rhif y dudalen", "nifer y dudalen", "rhif_nifer"),
        ("rhagor o wybodaeth", "mwy o wybodaeth", "mwy_rhagor"),
        ("mwy o geisiadau na llynedd", "rhagor o geisiadau na llynedd", "mwy_rhagor"),
        ("cymwysterau", "cymhwysterau", "sillafu"),
        ("cymwyseddau", "cymhwyseddau", "sillafu"),
    ]
    return choice_rows(
        "term", pairs,
        "Pa derm neu ymadrodd sy'n cyfateb i eirfa safonol y prosiect?",
    )


def build_summaries() -> list[dict[str, object]]:
    entities = [
        "Canolfan Iaith y Gogledd", "Labordy Llais Cymru", "Tîm Cyfieithu'r Brifysgol",
        "Prosiect Geirfa Agored", "Rhwydwaith Addysg Ddigidol", "Uned Ddata'r Coleg",
        "Canolfan Technoleg Iaith", "Partneriaeth Dysgu Cymraeg",
    ]
    actions = [
        "cyhoeddodd ganlyniadau peilot adnabod lleferydd",
        "cyhoeddodd ganlyniadau prawf cyfieithu",
        "cyflwynodd adroddiad ar grynhoi awtomatig",
        "lansiodd gorpws ymchwil newydd",
        "cwblhaodd arolwg o ddefnyddwyr",
        "cyhoeddodd archwiliad o ansawdd data",
        "profodd fodel iaith newydd",
        "cyflwynodd werthusiad o adnodd dysgu",
    ]
    dates = ["3 Mawrth", "18 Ebrill", "7 Mai", "21 Mehefin", "9 Gorffennaf", "14 Awst", "2 Medi", "27 Hydref"]
    numbers = [120, 84, 156, 210, 96, 175, 132, 248]
    percentages = [72, 68, 81, 64, 77, 59, 74, 83]
    next_steps = [
        "cyhoeddi'r data crai ym mis Medi",
        "ailadrodd y prawf gyda sampl fwy",
        "rhannu'r cod dan drwydded agored",
        "ychwanegu recordiadau o'r de",
        "cynnal gweithdy cyhoeddus yn yr hydref",
        "cywiro'r metadata cyn y lansiad",
        "profi'r model mewn ysgolion gwledig",
        "cyhoeddi'r fethodoleg lawn yr wythnos nesaf",
    ]
    rows: list[dict[str, object]] = []
    for index in range(24):
        base = index % 8
        cycle = index // 8
        entity = entities[base]
        action = actions[base]
        date = dates[(base + cycle) % 8]
        number = numbers[base] + 10 * cycle
        percentage = percentages[base] - cycle
        next_step = next_steps[(base + cycle) % 8]
        quote = f"Byddwn yn {next_step}."
        verb, object_phrase = action.split(maxsplit=1)
        source = (
            f"Ar {date}, {verb} {entity} {object_phrase}. Cymerodd {number} o bobl ran yn yr astudiaeth. "
            f"Roedd {percentage}% o'r ymatebwyr o blaid defnyddio'r gwasanaeth yn Gymraeg. "
            f"Dywedodd arweinydd y prosiect, “{quote}” Ni chafodd y gwaith ei ariannu gan gwmni preifat."
        )
        claims = (
            "F1: Cyhoeddwyd neu gyflwynwyd y gwaith ar y dyddiad a nodir.\n"
            "F2: Cymerodd y nifer a nodir o bobl ran.\n"
            "F3: Roedd y ganran a nodir o blaid y gwasanaeth Cymraeg.\n"
            "F4: Mae'r dyfyniad yn disgrifio'r cam nesaf.\n"
            "F5: Ariannwyd y gwaith gan gwmni preifat.\n"
            "F6: Roedd llai na hanner yr ymatebwyr o blaid y gwasanaeth Cymraeg."
        )
        system = (
            "Crynhoa'r darn mewn Cymraeg naturiol heb ychwanegu dim. Cadw'r dyfyniad yn union. "
            "Dychwela JSON yn unig: {\"crynodeb\": \"...\", \"ffeithiau\": [\"F1\"], "
            "\"dyfyniad\": \"...\"}. Rhestra bob gosodiad a gefnogir, a dim un arall."
        )
        rows.append(
            row(
                f"cryn-{index + 1:02d}", system,
                f"FFYNHONNELL:\n{source}\n\nGOSODIADAU:\n{claims}",
                {
                    "fact_ids": ["F1", "F2", "F3", "F4"],
                    "allowed_fact_ids": ["F1", "F2", "F3", "F4"],
                    "quote": quote,
                    "summary_concepts": [
                        [entity], [str(number)], [f"{percentage}%", f"{percentage} y cant"], [next_step],
                    ],
                },
                dimension="crynhoi_dyfynnu", synthetic_template=base + 1,
            )
        )
    return rows


def build_citations() -> list[dict[str, object]]:
    topics = [
        ("adnabod lleferydd", [("C1", "Cofnod y peilot Cymraeg", "12% oedd y gyfradd gwallau geiriau."), ("E1", "Speech pilot note", "The test used 40 hours of audio."), ("C2", "Nodyn am dafodieithoedd", "Perfformiodd y model orau ar leisiau o'r gogledd."), ("E2", "Deployment note", "No mobile test was conducted.")]),
        ("cyfieithu", [("C1", "Adroddiad cyfieithu", "Defnyddiwyd 1,500 o frawddegau yn y prawf."), ("E1", "Translation evaluation", "BLEU was reported separately."), ("C2", "Nodyn ansawdd", "Adolygodd dau gyfieithydd y sampl."), ("E2", "Cost note", "The run cost 18 dollars.")]),
        ("crynhoi", [("C1", "Prawf crynhoi", "Collodd y model dri ffaith bwysig."), ("E1", "Summary trial", "Quotations were checked by hand."), ("C2", "Canllaw'r prawf", "Y terfyn oedd 80 gair."), ("E2", "Model note", "Two models completed every case.")]),
        ("cyfeiriadau", [("C1", "Archwiliad ffynonellau", "Roedd dwy o'r pum ffynhonnell heb eu cadarnhau."), ("E1", "Citation audit", "Every title was searched in two catalogues."), ("C2", "Protocol gwirio", "Cofnodwyd DOI pob erthygl ddilys."), ("E2", "Reviewer note", "One reviewer repeated the audit.")]),
        ("cyfarwyddiadau", [("C1", "Arbrawf iaith", "Rhoddodd y cyfarwyddyd Cymraeg fwy o bwyslais ar ffynonellau Cymraeg."), ("E1", "Prompt study", "The interface language had little effect."), ("C2", "Nodyn ymholiad", "Defnyddiwyd y Gymraeg yn unig yn yr ymholiad olaf."), ("E2", "Search log", "The catalogue contained bilingual metadata.")]),
        ("cofrestrau", [("C1", "Prawf cywair", "Roedd y testun ffurfiol yn osgoi berfau ymadroddol."), ("E1", "Register test", "The informal sample contained dialogue."), ("C2", "Canllaw arddull", "Defnyddiwyd brawddegau gweithredol byr."), ("E2", "Scoring note", "There were 60 paired examples.")]),
        ("termau", [("C1", "Geirfa'r prosiect", "Sgwrsfot oedd y term am chatbot."), ("E1", "Terminology note", "The glossary contained 45 entries."), ("C2", "Cofnod termau", "Defnyddiwyd damcaniaethol yn lle theorïol."), ("E2", "Review note", "Three terms needed further discussion.")]),
        ("modelau agored", [("C1", "Cerdyn y model", "Cyhoeddwyd y pwysau dan drwydded agored."), ("E1", "Model card", "Training used 20 billion tokens."), ("C2", "Nodyn rhyddhau", "Roedd y cod gwerthuso ar GitHub."), ("E2", "Hardware note", "The run used four accelerators.")]),
    ]
    answer_concepts = [
        [[ ["12%", "12 y cant"] ], [ ["40 awr", "40 hours"] ], [ ["leisiau o'r gogledd", "northern voices"] ]],
        [[ ["1,500"] ], [ ["BLEU"], ["ar wahân", "separately"] ], [ ["dau gyfieithydd", "two translators"] ]],
        [[ ["dri ffaith", "three facts"] ], [ ["dyfyniadau", "quotations"], ["â llaw", "by hand"] ], [ ["80 gair", "80 words"] ]],
        [[ ["dwy o'r pum", "two of the five"] ], [ ["dau gatalog", "two catalogues"] ], [ ["DOI"] ]],
        [[ ["fwy o bwyslais", "more emphasis"], ["ffynonellau Cymraeg", "Welsh sources"] ], [ ["iaith y rhyngwyneb", "interface language"], ["fawr o effaith", "little effect"] ], [ ["Gymraeg yn unig", "Welsh only"] ]],
        [[ ["osgoi berfau ymadroddol", "avoided phrasal verbs"] ], [ ["deialog", "dialogue"] ], [ ["brawddegau gweithredol byr", "short active sentences"] ]],
        [[ ["sgwrsfot"] ], [ ["45"] ], [ ["damcaniaethol"], ["theorïol"] ]],
        [[ ["trwydded agored", "open licence"] ], [ ["20 biliwn", "20 billion"] ], [ ["GitHub"] ]],
    ]
    rows: list[dict[str, object]] = []
    for topic_index, (topic, sources) in enumerate(topics, 1):
        catalog = "\n".join(f"[{sid}] {title}: {statement}" for sid, title, statement in sources)
        allowed = [sid for sid, _, _ in sources]
        concepts = answer_concepts[topic_index - 1]
        questions = [
            (f"Beth mae'r ffynhonnell Gymraeg gyntaf yn ei ddweud am {topic}?", [sources[0][0]], concepts[0]),
            (f"Pa fanylyn sydd yn y ffynhonnell Saesneg gyntaf am {topic}?", [sources[1][0]], concepts[1]),
            (f"Beth mae'r ail ffynhonnell Gymraeg yn ei nodi am {topic}?", [sources[2][0]], concepts[2]),
            (f"Pwy oedd prif awdur yr ymchwil ar {topic}, a beth oedd rhif ei grant?", [], [["dim tystiolaeth", "nid oes digon o wybodaeth", "heb ei nodi"]]),
        ]
        for question_index, (question, source_ids, concepts) in enumerate(questions, 1):
            system = (
                "Ateba ar sail y catalog caeedig yn unig. Paid â dyfeisio ffynhonnell na manylyn. "
                "Os nad oes tystiolaeth, dywed hynny. Dychwela JSON yn unig: "
                "{\"ateb\": \"...\", \"cyfeiriadau\": [\"C1\"]}."
            )
            rows.append(
                row(
                    f"cyf-{topic_index:02d}-{question_index}", system,
                    f"CATALOG:\n{catalog}\n\nCWESTIWN:\n{question}",
                    {
                        "source_ids": source_ids,
                        "allowed_source_ids": allowed,
                        "answer_concepts": concepts,
                    },
                    dimension="cyfeiriadau", topic=topic,
                    answerable=bool(source_ids),
                )
            )
    assert len(rows) == 32
    return rows


def build_instruction_pair() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    topics = [
        ("adnabod lleferydd", "cyfradd gwallau", "tafodieithoedd", "speech recognition", "error rate", "dialects"),
        ("cyfieithu peirianyddol", "sgôr BLEU", "adolygu dynol", "machine translation", "BLEU score", "human review"),
        ("crynhoi awtomatig", "colli ffeithiau", "cadw dyfyniadau", "automatic summarisation", "omitted facts", "quotation preservation"),
        ("termau AI", "geirfa safonol", "bathiadau", "AI terminology", "standard vocabulary", "coinages"),
        ("modelau agored", "trwydded y pwysau", "data hyfforddi", "open models", "weights licence", "training data"),
        ("addysg uwch", "uniondeb academaidd", "llythrennedd AI", "higher education", "academic integrity", "AI literacy"),
        ("technoleg llais", "testun-i-leferydd", "lleferydd-i-destun", "voice technology", "text to speech", "speech to text"),
        ("cofrestrau iaith", "Cymraeg ffurfiol", "Cymraeg llafar", "language registers", "formal Welsh", "spoken Welsh"),
    ]
    welsh_rows: list[dict[str, object]] = []
    english_rows: list[dict[str, object]] = []
    for index, (topic, aspect_a, aspect_b, en_topic, en_aspect_a, en_aspect_b) in enumerate(topics, 1):
        sources = [
            (f"CY{index}A", "CY", f"Astudiaeth Gymraeg ar {topic}: {aspect_a} a {aspect_b}."),
            (f"CY{index}B", "CY", f"Adroddiad Cymraeg arall ar {topic}: {aspect_a}."),
            (f"EN{index}A", "EN", f"English study of {en_topic}: {en_aspect_a} and {en_aspect_b}."),
            (f"EN{index}B", "EN", f"English overview of {en_topic}: {en_aspect_a}."),
            (f"CY{index}X", "CY", "Erthygl am bwnc cwbl wahanol."),
        ]
        catalog = "\n".join(f"[{sid}] [{language}] {description}" for sid, language, description in sources)
        relevant_all = [f"CY{index}A", f"CY{index}B", f"EN{index}A", f"EN{index}B"]
        relevant_cy = [f"CY{index}A", f"CY{index}B"]
        variants = [
            (
                "rheolaeth",
                f"Dewisa bob ffynhonnell berthnasol am {topic}.",
                f"Select every relevant source about {topic}.",
                relevant_all,
            ),
            (
                "cymraeg_yn_unig",
                f"Dewisa bob ffynhonnell berthnasol am {topic}. Defnyddia ffynonellau Cymraeg yn unig.",
                f"Select every relevant source about {topic}. Use Welsh-language sources only.",
                relevant_cy,
            ),
            (
                "brawddeg_ccc",
                f"Dewisa bob ffynhonnell berthnasol am {topic}. Creu'r ymholiad gyda'r Gymraeg yn unig.",
                f"Select every relevant source about {topic}. Creu'r ymholiad gyda'r Gymraeg yn unig.",
                relevant_cy,
            ),
        ]
        for variant_index, (variant, cy_instruction, en_instruction, wanted) in enumerate(variants, 1):
            ideal = {"source_ids": wanted, "allowed_source_ids": wanted}
            cy_system = (
                "Dilyn y cyfarwyddyd a dewis o'r catalog yn unig. Dychwela JSON yn unig: "
                "{\"ffynonellau\": [\"CY1A\"]}."
            )
            en_system = (
                "Follow the instruction and select only from the catalogue. Return JSON only: "
                "{\"ffynonellau\": [\"CY1A\"]}."
            )
            welsh_rows.append(
                row(
                    f"cyf-cy-{index:02d}-{variant_index}", cy_system,
                    f"CATALOG:\n{catalog}\n\nCYFARWYDDYD:\n{cy_instruction}", ideal,
                    dimension="iaith_cyfarwyddyd", prompt_language="cy", variant=variant, topic=topic,
                )
            )
            english_rows.append(
                row(
                    f"cyf-en-{index:02d}-{variant_index}", en_system,
                    f"CATALOGUE:\n{catalog}\n\nINSTRUCTION:\n{en_instruction}", ideal,
                    dimension="iaith_cyfarwyddyd", prompt_language="en", variant=variant, topic=topic,
                )
            )
    assert len(welsh_rows) == len(english_rows) == 24
    return welsh_rows, english_rows


def write_jsonl(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    path = OUT / name
    content = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows)
    path.write_text(content, encoding="utf-8")
    return {
        "file": name,
        "cases": len(rows),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    welsh_instructions, english_instructions = build_instruction_pair()
    datasets = {
        "cywiro-esbonio.jsonl": build_editing(),
        "priod-ddull.jsonl": build_idiom(),
        "termau.jsonl": build_terms(),
        "crynhoi-dyfynnu.jsonl": build_summaries(),
        "cyfeiriadau.jsonl": build_citations(),
        "cyfarwyddyd-cymraeg.jsonl": welsh_instructions,
        "cyfarwyddyd-saesneg.jsonl": english_instructions,
    }
    files = [write_jsonl(name, rows) for name, rows in datasets.items()]
    manifest = {
        "name": "Archwiliad Iaith CCC",
        "version": "0.1.0-beta",
        "total_cases": sum(item["cases"] for item in files),
        "files": files,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
