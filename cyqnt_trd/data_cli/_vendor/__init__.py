"""Vendored, dependency-free third-party clients used by ``data_cli``.

Everything under ``_vendor`` is a *pure-stdlib* copy of upstream code, pinned
here so ``cyqnt_trd`` never grows a new runtime dependency and never reaches
into any private / internal network surface. See each module's
``__provenance__`` for the exact upstream source, revision, and the subset that
was intentionally vendored.
"""
