# Stage 4 OFI diagnostic — DOT-USD

book_change events: 45,789

Sign convention: corr>0 = OFI predicts CONTINUATION (trend → MM leans defensively); corr<0 = REVERSION (MM leans aggressively); ~0 = no exploitable signal.

| window | horizon | n | OFI Pearson | OFI Spearman | depth-imb Pearson | depth-imb Spearman |
|---|---|---|---|---|---|---|
| 5s | 5s | 23628 | -0.2045 | -0.2292 | -0.0320 | -0.0308 |
| 5s | 30s | 11150 | -0.1770 | -0.2040 | -0.0273 | -0.0297 |
| 30s | 5s | 23628 | -0.1424 | -0.1805 | -0.0320 | -0.0308 |
| 30s | 30s | 11150 | -0.1584 | -0.1807 | -0.0273 | -0.0297 |
