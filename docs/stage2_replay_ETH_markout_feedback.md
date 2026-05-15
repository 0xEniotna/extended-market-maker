# Markout-Feedback Replay — real implementation on ETH journals
Journals: 4
- `/root/MM/data/mm_journal/mm_ETH-USD_20260505_171314.jsonl`
- `/root/MM/data/mm_journal/mm_ETH-USD_20260506_102154.jsonl`
- `/root/MM/data/mm_journal/mm_ETH-USD_20260506_102154.1.jsonl`
- `/root/MM/data/mm_journal/mm_ETH-USD_20260506_102154.2.jsonl`

## Combos
| combo | params | n | %active | mean_widen | max_widen | cap viol | markout(act) | markout(inact) | diff |
|---|---|---|---|---|---|---|---|---|---|
| recommended | hl=30s th=2.0 g=0.5 cap=5.0 | 1751 | 26.0% | 3.45 | 5.00 | 0 | -2.557 | -2.278 | -0.278 |
| aggressive | hl=30s th=1.0 g=1.0 cap=10.0 | 1751 | 34.6% | 6.12 | 10.00 | 0 | -2.441 | -2.303 | -0.139 |
| conservative | hl=60s th=2.0 g=0.5 cap=5.0 | 1751 | 32.6% | 3.81 | 5.00 | 0 | -2.494 | -2.281 | -0.213 |

## Verification gates
- ✅ No cap violations across all combos = implementation respects bound
- ✅ diff < -1.0 for recommended combo = real code matches calibration
- ✅ %active in 30-50% range = expected

If any of these is ❌, do NOT launch the iter — investigate.
