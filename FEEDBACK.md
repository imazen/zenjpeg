# User Feedback Log

## 2026-01-31
- User request: Fix pre-erosion lookahead timing (C++ has 4-row overlap at iMCU boundaries)
- Investigation found root cause was v_samp=1 for XYB AQ instead of v_samp=2
- User noted concern about memory usage from large buffer dumps, suggested expanding rotating buffer instead
