# Paper results and release scope

The current manuscript reports five-domain complete-system results, paired component ablations, frozen identity diagnostics, and transfer across visual feature extractors. The README separates designated main-table runs from multi-seed paired summaries. Reproduction commands and dependencies are documented in [REPRODUCING.md](REPRODUCING.md).

## Additional OWL-ViT L/14 rows in Table 3

The following mean ± SD values were supplied by the author for the updated manuscript. Columns are K/U AP50 (%). No per-seed records, exact D2/D3 module definitions, or matching L/14 reproduction configuration were supplied with these rows; the existing `configs/owl_reference.json` implements the B/16 O0/O1 study. These additions do not establish a nine-pair L/14 claim.

| Domain | L/14 D2 K | L/14 D2 U | L/14 D3 K | L/14 D3 U |
|---|---:|---:|---:|---:|
| Medical | 53.13 ± 0.11 | 7.09 ± 0.31 | 65.74 ± 0.32 | 10.83 ± 0.44 |
| Aquatic | 62.85 ± 0.05 | 19.33 ± 0.24 | 63.85 ± 0.11 | 21.20 ± 0.04 |
| Game | 28.37 ± 0.21 | 1.61 ± 0.06 | 68.37 ± 1.24 | 33.31 ± 1.05 |

## Interpretation and availability

Task 2 is joint fitting with previously and currently known labels. The downstream visual pipeline does not imply that foundation pre-training excluded every test concept. Unknown identity scores are not guaranteed calibrated probabilities. Medical and surgical results are benchmark evaluations, without a clinical deployment claim.

The release preserves source code, configurations, protocols, lightweight results, and checkpoint download interfaces. See [CHECKPOINTS.md](CHECKPOINTS.md) for current model availability. Full original prediction archives and all historical experiment workspaces are not committed. Runtime tooling does not itself constitute measured latency results; the current manuscript makes no speed-advantage claim. Portable-interface GPU execution remains subject to the release validation status described in the README.
