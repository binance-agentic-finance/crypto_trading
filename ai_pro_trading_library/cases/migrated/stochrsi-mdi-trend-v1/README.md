# stochrsi-mdi-trend-v1

Trend-following case originally named in the priority list. The original
source case was not located in the first inventory pass; the migrated bundle
is a placeholder with documented intentional divergence.

- **Status**: bootstrap. Listed in `catalog.json` as `not_executable_yet`
  pending source recovery and parity verification.
- **Signal**: `signal.py::build_signal` is a SMA10 > SMA20 long stand-in.
- **Next step**: locate or rebuild the original StochRSI / MDI trend logic and
  promote to `migrated`.
