"""Repo-level tooling namespace.

Deliberately empty of imports: a package that pulled its subpackages in at
import time would make ``import tools.nl2yaml.capability`` depend on every other
tool's dependencies being installed.
"""
