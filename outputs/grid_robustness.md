# Robustness across the 16-condition prompt grid

Per cell: coverage (model x dataset units published) and, over the covered units, mean F1 and MCC at tau=0 and the mean oracle-MCC gain.

Coverage rationale: the paper's evidence rests entirely on the primary condition (complete, 32/32). The wider grid is a robustness check, run in full on OC3-FO (the hardest benchmark) and for the lighter half of the grid on the three larger datasets; the chain-of-thought conditions proved ~neutral on OC3-FO and were not extended further. Cells covering only OC3-FO show higher mean oracle gains simply because threshold gains are largest there.

| Cell | Coverage | mean F1@0 | mean MCC@0 | mean ΔMCC(oracle−τ0) |
|---|---|---|---|---|
| valsoff__shot0__nocot__t0.0__sc1 | 32/32 | 0.798 | 0.713 | +0.063 |
| valson__shot0__nocot__t0.0__sc1 | 32/32 | 0.799 | 0.706 | +0.053 |
| valsoff__shot3__nocot__t0.0__sc1 | 32/32 | 0.775 | 0.668 | +0.074 |
| valson__shot3__nocot__t0.0__sc1 | 32/32 | 0.746 | 0.656 | +0.051 |
| valsoff__shot0__cot__t0.0__sc1 | 8/32 | 0.576 | 0.542 | +0.097 |
| valson__shot0__cot__t0.0__sc1 | 8/32 | 0.575 | 0.544 | +0.102 |
| valsoff__shot3__cot__t0.0__sc1 | 8/32 | 0.510 | 0.470 | +0.074 |
| valson__shot3__cot__t0.0__sc1 | 8/32 | 0.536 | 0.500 | +0.046 |
| valsoff__shot0__nocot__t0.7__sc3 | 32/32 | 0.816 | 0.741 | +0.039 |
| valson__shot0__nocot__t0.7__sc3 | 16/32 | 0.732 | 0.701 | +0.039 |
| valsoff__shot3__nocot__t0.7__sc3 | 15/32 | 0.642 | 0.606 | +0.052 |
| valson__shot3__nocot__t0.7__sc3 | 32/32 | 0.758 | 0.681 | +0.042 |
| valsoff__shot0__cot__t0.7__sc3 | 8/32 | 0.665 | 0.641 | +0.046 |
| valson__shot0__cot__t0.7__sc3 | 8/32 | 0.629 | 0.601 | +0.043 |
| valsoff__shot3__cot__t0.7__sc3 | 8/32 | 0.511 | 0.490 | +0.054 |
| valson__shot3__cot__t0.7__sc3 | 8/32 | 0.528 | 0.503 | +0.037 |
