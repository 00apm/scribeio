# scribe

Small local tool for TTRPG sessions recorded with Craig (Discord).
One Craig `.zip` in → a mastered audio mix for the players and a transcript
ready for an LLM assistant. Personal use only.

```
scribe master     <zip>   # mastered mix (mp3)
scribe transcribe <zip>   # transcript .md + .json + quality report
```

## Install (conda env `scribe`)

torch (cu128) and WhisperX are installed into the env first; then:

```
pip install -e .[dev]
pytest -m "not gpu"
```

## Layout

```
src/scribe/     the package
tests/          pytest
```
