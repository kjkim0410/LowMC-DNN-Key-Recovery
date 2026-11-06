# LowMC DNN Key Recovery — ReLU Equivalence and Internal Activation Leakage

Reproducible Python research code for **LowMC-L1, LowMC-L3 and LowMC-L5**. The repository collects three stages of experiments: the original v5 synthetic-oracle key-recovery code, v6 instrumented ReLU implementations, and v7 recovery of the first-round key-schedule output (RK1) from selected internal activations followed by master-key reconstruction using the inverse public key-schedule matrix.

## Scope and limitations

**Important:** v7 assumes direct access to both post-ReLU outputs of the final XOR gate for **every RK1 coordinate**. This is a strong *white-box internal-node observation model*, not a demonstrated physical side channel, limited-observation attack, or black-box attack on Picnic. The provided Boolean/ReLU equivalence tests apply to binary inputs. The experiments are educational/research proof-of-concept code, not a claim of practical compromise of deployed cryptography.

## Repository layout

```text
LowMC-DNN-Key-Recovery/
├── README.md
├── .gitignore
├── lowmc_L1_package/
│   ├── README.md
│   ├── lowmc_L1_grain_keyrecovery_v5_1000.py
│   ├── lowmc_L1_grain_relu_instrumented_v6.py
│   ├── lowmc_L1_activation_recovery_v7.py
│   └── results/
├── lowmc_L3_package/       (same structure)
└── lowmc_L5_package/       (same structure)
```

## Requirements

- Python **3.10** (standard library only for the v7 runners)
- No third-party packages required for the v7 campaigns
- Tested campaign results are supplied in each `results/` directory

## Quick start

From the repository root (PowerShell):

```powershell
cd .\lowmc_L1_package
python .\lowmc_L1_activation_recovery_v7.py --mode all --trials 10
```

For L3 and L5, replace `L1` in the directory and script names. The `--mode` option accepts `equiv`, `recovery`, or `all`; `--trials` accepts one or more integers. To run all included campaign sizes:

```powershell
python .\lowmc_L1_activation_recovery_v7.py --mode all --trials 10 1000
```

The 1,000-trial full-equivalence runs can be slow, especially for L5. Each package's `README.md` explains its programs and commands.

## Experimental results supplied with the archive

| Level | Block/key bits | Rounds | Equivalence (10 / 1000) | RK1 + master recovery (10 / 1000) | Internal ReLU values per recovery trial |
|---|---:|---:|---|---|---:|
| L1 | 128 | 20 | 10/10 · 1000/1000 | 10/10 · 1000/1000 | 256 |
| L3 | 192 | 30 | 10/10 · 1000/1000 | 10/10 · 1000/1000 | 384 |
| L5 | 256 | 38 | 10/10 · 1000/1000 | 10/10 · 1000/1000 | 512 |

The table reports the `summary.json` files included in the user-provided archive; these are **not independent replications**. Full per-trial CSVs are retained. The recorded maximum master-key Hamming distance is zero for all supplied recovery campaigns.

## Method at a glance

1. Generate LowMC public matrices using Grain-SSG.
2. Compare Boolean and ReLU realizations on binary test vectors.
3. Instrument the final ReLU XOR nodes for the first-round key-schedule output RK1.
4. Reconstruct RK1 from the observed activation pairs.
5. Invert the public first-round key-schedule matrix and compare the recovered master key with the test key.

## Reproducibility notes

- v7 outputs are written to each package's `results/` directory.
- Existing results are preserved; rerunning a campaign can overwrite its corresponding CSV and update its summary.
- The provided results are simulation outputs under an explicitly instrumented oracle, not measurements from real devices.
- For scholarly citation, please add the associated paper's title, authors, DOI or ePrint identifier once finalized.

## License

No open-source license is assigned in this archive. The repository owner should add an appropriate `LICENSE` file before inviting reuse or redistribution.
