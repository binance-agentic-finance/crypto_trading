"""Natural-language trading request -> YAML strategy spec, and the gates on it.

Imports nothing on purpose. Several independent pieces land in this package
(capability table, gates, corpus tooling, converter), and a package __init__
that re-exported them would couple every piece to every other one's import
graph — so a corpus script could not run without pandas being importable, and a
capability lookup could not run without the standard_bot stack.
"""
