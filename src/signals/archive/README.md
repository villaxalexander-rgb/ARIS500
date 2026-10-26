# Archived signal blocks

These modules are research records, not production model inputs. They remain
importable so the signal-research notebook can reproduce the tests and so a
future contributor can see what failed before proposing the same idea again.

- `risk.py`: low volatility, low beta, and low idiosyncratic volatility had
  inverted quintiles and a negative long-short spread. The live portfolio's
  beta estimator is separate in `src/signals/beta.py`; archiving the rejected
  block does not disable the portfolio beta constraint.
- `fundamental_momentum.py`: CFO and revenue trends weakened as the fixed
  lookback lengthened and were essentially absent at the chosen 12-quarter
  robustness window. It was excluded rather than tuned to the best window.

Archived blocks are registered in `src/signals/registry.py` with
`archived=True` and `live=False`. Do not add one to `config.MODEL_BLOCKS`
without a new documented research decision.
