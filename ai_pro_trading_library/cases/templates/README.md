# Case Template

Each migrated case directory must contain:

- `case.json` — case metadata; validated against `case.schema.json`.
- `preset.json` — preferred parameter preset.
- `strategy.py` — `build_spec()` returning `StrategySpec`.
- `signal.py` — `build_signal(spec, data)` returning a long-signal `pd.Series`.
- `README.md` — short description and source attribution.

Drop a fresh case directory next to `migrated/` and run the catalog tests to
register the new signal builder automatically.
