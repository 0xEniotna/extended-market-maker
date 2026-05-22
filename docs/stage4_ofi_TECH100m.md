# Stage 4 OFI diagnostic — TECH100m-USD

book_change events: 84,031

Sign convention: corr>0 = OFI predicts CONTINUATION (trend → MM leans defensively); corr<0 = REVERSION (MM leans aggressively); ~0 = no exploitable signal.

| window | horizon | n | OFI Pearson | OFI Spearman | depth-imb Pearson | depth-imb Spearman |
|---|---|---|---|---|---|---|
| 5s | 5s | 22110 | -0.0390 | -0.1087 | +0.0552 | +0.0816 |
| 5s | 30s | 9040 | -0.0465 | -0.0984 | +0.0478 | +0.0501 |
| 30s | 5s | 22110 | -0.0094 | -0.0464 | +0.0552 | +0.0816 |
| 30s | 30s | 9040 | -0.0092 | -0.0700 | +0.0478 | +0.0501 |
