# Data CymraegBench Voice v0.1

Mae `tts-prompts.jsonl` yn set CC0 a ysgrifennwyd ar gyfer y meincnod hwn. Mae'n
cynnwys sgwrs, enwau lleoedd, treigladau, rhifau, dyddiadau, talfyriadau,
termau technegol, cyfnewid cod ac atalnodi.

Nid yw sain ASR yn cael ei chadw yn Git. Rhaid creu
`data/private/voice/arfor-test-clean.jsonl`; mae pob rhes yn defnyddio'r ffurf:

```json
{"id":"arfor-test-000000","audio":"audio/arfor-test-000000.wav","reference":"O MAE'N","metadata":{"accent":"Gogledd Orllewin"}}
```

Mae llwybrau `audio` yn gymharol i'r manifest. Rhaid cadw setiau profi ar wahân
i ddata hyfforddi'r model, a chofnodi revision a thrwydded y ffynhonnell yn
`config/voice-suites.toml`.

Mae'r set TTS yn mesur llwyddiant, latency, RTF, clipping a distawrwydd yn
awtomatig. Dylid defnyddio'r union WAVs a grëir mewn prawf gwrando dall ar
gyfer dealladwyedd, naturioldeb ac ynganu; nid yw metrig awtomatig yn cymryd
lle barn siaradwyr Cymraeg.
