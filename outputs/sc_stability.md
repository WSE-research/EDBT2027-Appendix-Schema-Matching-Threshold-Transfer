# Self-consistency score stability (cell valsoff__shot0__nocot__t0.7__sc3__scopeoff, 8 models)

Identical-decision cases = (task, source) pairs where all three temp-0.7 runs choose the same target. The last column reads the spread only among confidence-10 cases (mean score ≥ 9.5).

| Dataset | N identical-decision | % identical score | mean spread | median | P90 | mean spread @conf10 |
|---|---|---|---|---|---|---|
| ppm | 1347 | 74.2% | 0.33 | 0.0 | 1.0 | 0.075 (n=871) |
| val | 62448 | 89.3% | 0.17 | 0.0 | 0.5 | 0.033 (n=55300) |
| hdx | 14573 | 87.7% | 0.17 | 0.0 | 1.0 | 0.037 (n=12429) |
| oc3 | 222 | 71.2% | 0.41 | 0.0 | 1.0 | 0.099 (n=126) |

Pooled: N=78590, identical score in all three runs = 88.7%, mean spread 0.17 points.
