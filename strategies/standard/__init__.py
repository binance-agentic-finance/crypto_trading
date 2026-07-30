"""Sample bots on the StandardBot contract.

Ported from the funding-based books validated in the local strategy workspace
(``策略开发/40_A_carry``, ``50_N_neutral``, ``70_D_derivatives``). Those are the
strategies that actually held up out-of-sample, so they are the honest thing to
port: every one of them is a funding read, and funding is the single deepest
replayable series available.
"""
