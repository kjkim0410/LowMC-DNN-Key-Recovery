# LowMC-L5: ReLU equivalence and activation-based recovery

Parameters used in this package: 256-bit block/key, 38 rounds, 10 S-boxes.

## Programs

- `lowmc_L5_grain_keyrecovery_v5_1000.py`: original v5 synthetic-oracle recovery experiment.
- `lowmc_L5_grain_relu_instrumented_v6.py`: instrumented Boolean/ReLU implementation.
- `lowmc_L5_activation_recovery_v7.py`: v7 equivalence and activation-recovery campaigns.

## Run (Python 3.10, standard library)

From this directory:

```powershell
python .\lowmc_L5_activation_recovery_v7.py --mode all --trials 10
python .\lowmc_L5_activation_recovery_v7.py --mode recovery --trials 1000
python .\lowmc_L5_activation_recovery_v7.py --mode equiv --trials 1000
```

Use `--mode all --trials 10 1000` to run the complete campaigns. `--seed` sets reproducible pseudorandom inputs. Full 1000-trial equivalence may take considerable time.

## Included experimental results

| Campaign | Trials | Successful | Seconds |
|---|---:|---:|---:|
| equivalence | 10 | 10/10 | 21.8892 |
| activation_recovery | 10 | 10/10 | 0.5448 |
| equivalence | 1000 | 1000/1000 | 2120.264 |
| activation_recovery | 1000 | 1000/1000 | 45.1661 |

Results are stored in `results/` (CSV and JSON). These are **supplied results**, not independently rerun benchmark measurements in this repository-preparation step.

## Threat model

v7 observes both post-ReLU outputs of the final XOR for every coordinate of the first-round key-schedule output (RK1), amounting to **512 internal activation values per trial**. RK1 is reconstructed and the public invertible key-schedule matrix is inverted to recover the master key. This is an explicit, strong white-box instrumentation model. It does **not** demonstrate exploitable noisy hardware leakage, a black-box attack, or a practical attack on Picnic. Boolean/ReLU equivalence is tested on binary inputs.
