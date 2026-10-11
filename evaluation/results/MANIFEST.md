# Benchmark result manifest

Result files are git-ignored (they are large and contain TriviaQA questions and answers). This manifest records
every run used in the documentation so any copy can be verified with `Get-FileHash -Algorithm SHA256`.
A local archive was created at `data/runtime/archive/benchmark_results_2026-10-11.zip`.

Method names: until routing became the default, `adaptive` meant v2 with routing off (now the variant `adaptive_v2`).

Note: runs before 2026-10-11 16:00 recorded the git commit when the file was *saved*; the commit that actually
executed is given in `docs/results.md` (Part C: `808ef33`). Later runs record the commit at start.

| Run | Doc | Purpose | Dataset | Systems | Recorded commit | SHA-256 results.json | SHA-256 results.csv |
|---|---|---|---|---|---|---|---|
| `20261010-212100-3c2e8d` | Part A | smoke test, original v1 | smoke_test (24 q) | standard, adaptive | n/a | `e25949def1bedf55c1a5553c676d361f7eb5e3983155d8526fb3b3b03478a546` | `1471c52370a0123220e2e74cd803d65642c67a9721e64aa05aaba4a65bd2d159` |
| `20261010-223619-6d2293` | Part A | original baseline: Standard vs Multi-Agent v1, dev set | triviaqa_unfiltered_sample (100 q) | standard, adaptive | n/a | `7b8cbd2bfc3619e812e433fc7c65def9833a2ab451441a2c730b60b1c78a3635` | `2716c5b2137594c6ab4c5ec92e65d93628424f9c82f4bbc46102f2897719b3f9` |
| `20261011-110725-f69cff` | Part B | v2 development iteration 1 (dev set) | triviaqa_unfiltered_sample (100 q) | adaptive | 076c950-dirty | `b4a83e52e88a532c10121bfa0d1ecde4f27872763394e07bce2dfa868d2a6bb6` | `d794bd9402f96beed52d0650efce5f6727847e900f075c3d95e64c68fbe07026` |
| `20261011-115334-b96d4c` | Part B | v2 development iteration 2 (dev set) | triviaqa_unfiltered_sample (100 q) | adaptive | 52a4dc4-dirty | `8e9f4085e2623972c0c74bfd4b635846820d23e645874246d06ec76771aa9093` | `623f0302a57d4d48d938705671d5189ed05646335633a98e95b8b5e882568af2` |
| `20261011-122005-0a2d3c` | Part C | held-out test of frozen v2 + ablations (executed code 808ef33) | triviaqa_heldout_101_300 (199 q) | standard, adaptive, adaptive_no_retry, adaptive_no_filter, adaptive_v1 | a703ed7 | `5420ebe725c1465c90274b0a7f520d1b0092192d3572de3ff77de66e04e0e1ff` | `5b50785684de857555c31a6e74fd4e9b4e54a3f099865d640785af5ac2b2df07` |
| `20261011-163907-4a3df8` | audit | smoke test before merge (v2) | smoke_test (24 q) | standard, adaptive | 9aa9d89 | `3bcbe3ec33a3815e81555559d74f5e2941ae39f2c330f8d36cb3415259b40d25` | `67a003a905ff0093772a6a504464ff62654c1d2b33382ffd29928d80f12f3c22` |
| `20261011-165516-f102b5` | Part D dev | question routing on dev set | triviaqa_unfiltered_sample (100 q) | adaptive_routed | 57aef61 | `5b61c24ea6d940db797df56b976deda326d9ecbdc70f4838c0d65435748781ea` | `d51ab4650ffe5da369024b710bd07d86efb6be8517242857ed152fc96c83b54b` |
| `20261011-171730-9902e7` | Part D | held-out re-run: Standard vs v2 vs v2+routing | triviaqa_heldout_101_300 (199 q) | standard, adaptive, adaptive_routed | 57aef61 | `8ec2985f7d0261ac75f0870c381e7eac3c9c61d1ae3be04d2b0bab9093f34931` | `b8ed7d6aec1a0709018e29633fb37a3106947c93c7df1ae679316092748e8cfd` |
| `20261011-193700-e448b5` | audit | pre-merge smoke test with routing on by default | smoke_test (24 q) | standard, adaptive | 02c78b4-dirty | `c8b3ceeb044950aabf872c2362dbacefbbc1d3eb51227113f1b5ffb75b1e78fc` | `a52ddd3cae024712d498974f9e6a1849d914372c0c8adeabe6e5faa679d4fc68` |

`20261011-193700-e448b5` recorded `02c78b4-dirty`: it ran the uncommitted change that made routing the default, which was
committed unchanged as `ef88805` immediately afterwards.
