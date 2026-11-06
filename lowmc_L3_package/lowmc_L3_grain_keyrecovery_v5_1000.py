#!/usr/bin/env python3
"""
lowmc_L3_grain_keyrecovery_v5.py

Integrated LowMC-L3 Grain-SSG key schedule + noisy rk1 leakage experiment.

This file merges the roles of:
  1) lowmc_picnic_boolean_grain.py
     - Grain LFSR self-shrinking generator
     - LowMC-L3 public matrices/constants generation
     - K_0,...,K_30 generation
     - round-key relation RK_i = K_i * master_key over GF(2)

  2) lowmc_bitwise_keyrecovery_sym_change_v4.py
     - noisy threshold leakage oracle over rk1 bits
     - symmetric/change diagnostic pair tests
     - statistical rk1 recovery
     - master-key recovery from recovered rk1

Key difference from v4
----------------------
The old v4 used:

    self.K1 = random_invertible_binary_matrix(128)

This v5 removes that experimental random K1 completely.

Instead:

    Grain SSG
       -> L_1,...,L_30
       -> C_1,...,C_30
       -> K_0,...,K_30
       -> choose K_1
       -> RK_1 = K_1 * master_key
       -> noisy oracle
       -> recovered RK_1
       -> master_key = K_1^{-1} * RK_1

IMPORTANT
---------
This is a research/self-study experiment.  The leakage oracle is synthetic.
The LowMC-L3 key-matrix generation follows the Grain-SSG construction used
by the LowMC reference generator model used in the companion Boolean code.

Python 3.8+.
No NumPy/PyTorch required.
"""

import random
import time


# ============================================================
# LowMC-L3 parameters
# ============================================================

NBITS = 192
NKEY = 192
M_SBOX = 10
ROUNDS = 30

# Leakage parameters
MU = 0.35
NOISE_BOUND = 0.80
RK_RECOVERY_TRIALS = 401
PAIR_TRIALS = 200

RANDOM_SEED = 20260810


# ============================================================
# Small utilities
# ============================================================

def mask(width):
    return (1 << width) - 1


def parity(x):
    return bin(x).count("1") & 1


def rand_bits(n):
    return [random.getrandbits(1) for _ in range(n)]


def bits_to_hex(bits):
    n = 0
    for b in bits:
        n = (n << 1) | (b & 1)
    width = (len(bits) + 3) // 4
    return "0x{:0{}x}".format(n, width)


def int_to_bitlist_msb(x, n):
    """
    Convert integer to MSB-first list, matching the representation
    used by the original v4 leakage code.
    """
    return [((x >> (n - 1 - i)) & 1) for i in range(n)]


def bitlist_msb_to_int(bits):
    x = 0
    for b in bits:
        x = (x << 1) | (b & 1)
    return x


def hamming_distance(a, b):
    return sum((x ^ y) for x, y in zip(a, b))


# ============================================================
# Grain LFSR self-shrinking generator
# ============================================================

def grain_ssg():
    """
    Grain-LFSR-based self-shrinking generator used for deterministic
    LowMC public parameter generation.

    State:
        80 bits initialized to 1.

    First 160 update/output bits are discarded.

    Self-shrinking rule:
        selector = step()
        candidate = step()

        if selector == 1:
            output candidate
        else:
            discard candidate
    """
    state = [1] * 80
    index = 0

    def step():
        nonlocal index

        state[index] ^= (
            state[(index + 13) % 80]
            ^ state[(index + 23) % 80]
            ^ state[(index + 38) % 80]
            ^ state[(index + 51) % 80]
            ^ state[(index + 62) % 80]
        )

        out = state[index] & 1
        index = (index + 1) % 80
        return out

    for _ in range(160):
        step()

    while True:
        selector = step()
        candidate = step()

        if selector == 1:
            yield candidate


# ============================================================
# GF(2) integer-row matrix routines
# ============================================================

def random_row_from_generator(cols, gen):
    row = 0

    for j in range(cols):
        row |= (next(gen) & 1) << j

    return row


def gf2_rank_int(rows, cols):
    A = rows[:]
    rank = 0

    for col in range(cols):
        pivot = None

        for i in range(rank, len(A)):
            if (A[i] >> col) & 1:
                pivot = i
                break

        if pivot is None:
            continue

        A[rank], A[pivot] = A[pivot], A[rank]
        pivot_row = A[rank]

        for i in range(len(A)):
            if i != rank and ((A[i] >> col) & 1):
                A[i] ^= pivot_row

        rank += 1

        if rank == min(len(rows), cols):
            break

    return rank


def instantiate_matrix(nrows, ncols, gen):
    """
    Generate a full/maximal-rank binary matrix from Grain-SSG bits.

    If a whole candidate matrix is not full rank, discard it and
    continue consuming bits from the same Grain stream.
    """
    required_rank = min(nrows, ncols)

    while True:
        rows = [
            random_row_from_generator(ncols, gen)
            for _ in range(nrows)
        ]

        if gf2_rank_int(rows, ncols) == required_rank:
            return rows


def mat_vec_mul_int(rows, vector):
    """
    GF(2) matrix-vector multiplication using integer rows.

        y_i = parity(row_i & vector)
    """
    y = 0

    for i, row in enumerate(rows):
        y |= parity(row & vector) << i

    return y


def gf2_inverse_int(square_rows, n):
    if len(square_rows) != n:
        raise ValueError("Matrix must be square.")

    left = square_rows[:]
    right = [1 << i for i in range(n)]

    pivot_row = 0

    for col in range(n):
        pivot = None

        for r in range(pivot_row, n):
            if (left[r] >> col) & 1:
                pivot = r
                break

        if pivot is None:
            raise ValueError("Matrix is singular over GF(2).")

        left[pivot_row], left[pivot] = left[pivot], left[pivot_row]
        right[pivot_row], right[pivot] = right[pivot], right[pivot_row]

        for r in range(n):
            if r != pivot_row and ((left[r] >> col) & 1):
                left[r] ^= left[pivot_row]
                right[r] ^= right[pivot_row]

        pivot_row += 1

    return right


# ============================================================
# Conversion between LowMC integer-row matrices and v4 bit lists
# ============================================================

def int_matrix_to_msb_bit_matrix(rows, ncols):
    """
    Convert integer-row representation to MSB-first list-of-lists.

    Important:
    Integer row bit j corresponds to coordinate j in the companion
    Boolean implementation.  To preserve the matrix relation while using
    the old v4 MSB-first list representation, we reverse bit positions.
    """
    M = []

    for row in rows:
        bits = [((row >> (ncols - 1 - j)) & 1) for j in range(ncols)]
        M.append(bits)

    # Output coordinate ordering from mat_vec_mul_int is LSB-indexed.
    # Reverse rows too so that resulting list is MSB-first.
    return M[::-1]


def msb_bit_matrix_to_int_rows(M):
    """
    Inverse helper for diagnostics.
    """
    rows = []

    # reverse output-row order
    for row_bits in M[::-1]:
        x = 0
        ncols = len(row_bits)

        for j, b in enumerate(row_bits):
            if b:
                bitpos = ncols - 1 - j
                x |= 1 << bitpos

        rows.append(x)

    return rows


# ============================================================
# Original v4-style list GF(2) routines
# ============================================================

def dot_row_vec_mod2(row, vec):
    s = 0

    for x, y in zip(row, vec):
        s ^= (x & y)

    return s


def mat_vec_mul_gf2(M, v):
    return [dot_row_vec_mod2(row, v) for row in M]


def identity_matrix(n):
    I = []

    for i in range(n):
        row = [0] * n
        row[i] = 1
        I.append(row)

    return I


def copy_matrix(M):
    return [row[:] for row in M]


def gf2_rank(M):
    A = copy_matrix(M)
    nrows = len(A)
    ncols = len(A[0]) if nrows > 0 else 0
    r = 0

    for c in range(ncols):
        pivot = None

        for i in range(r, nrows):
            if A[i][c]:
                pivot = i
                break

        if pivot is None:
            continue

        A[r], A[pivot] = A[pivot], A[r]

        for i in range(nrows):
            if i != r and A[i][c]:
                for j in range(c, ncols):
                    A[i][j] ^= A[r][j]

        r += 1

        if r == nrows:
            break

    return r


def gf2_inverse(M):
    n = len(M)
    A = copy_matrix(M)
    Inv = identity_matrix(n)

    row = 0

    for col in range(n):
        pivot = None

        for r in range(row, n):
            if A[r][col]:
                pivot = r
                break

        if pivot is None:
            raise ValueError("Matrix is not invertible over GF(2).")

        if pivot != row:
            A[row], A[pivot] = A[pivot], A[row]
            Inv[row], Inv[pivot] = Inv[pivot], Inv[row]

        for r in range(n):
            if r != row and A[r][col]:
                for c in range(n):
                    A[r][c] ^= A[row][c]
                    Inv[r][c] ^= Inv[row][c]

        row += 1

    return Inv


# ============================================================
# LowMC-L3 public parameter generation
# ============================================================

class LowMCL3PublicParameters(object):
    """
    Deterministically generate:

        L_1,...,L_30
        C_1,...,C_30
        K_0,...,K_30

    from one Grain-SSG stream.
    """

    def __init__(self, verbose=False):
        self.n = NBITS
        self.k = NKEY
        self.m = M_SBOX
        self.r = ROUNDS

        gen = grain_ssg()

        if verbose:
            print("[Grain] generating L_1,...,L_30")

        self.linear_matrices = [
            instantiate_matrix(self.n, self.n, gen)
            for _ in range(self.r)
        ]

        if verbose:
            print("[Grain] generating C_1,...,C_30")

        self.round_constants = []

        for _ in range(self.r):
            c = 0

            for j in range(self.n):
                c |= (next(gen) & 1) << j

            self.round_constants.append(c)

        if verbose:
            print("[Grain] generating K_0,...,K_30")

        self.round_key_matrices = [
            instantiate_matrix(self.n, self.k, gen)
            for _ in range(self.r + 1)
        ]

        self.K1_int = self.round_key_matrices[1]

        if gf2_rank_int(self.K1_int, self.k) != self.k:
            raise AssertionError("Generated K1 is unexpectedly singular.")

        self.K1_inv_int = gf2_inverse_int(self.K1_int, self.n)

        # v4-compatible list representation
        self.K1_bits = int_matrix_to_msb_bit_matrix(
            self.K1_int, self.k
        )

        if gf2_rank(self.K1_bits) != self.k:
            raise AssertionError(
                "Converted MSB-bit K1 is unexpectedly singular."
            )

        self.K1_inv_bits = gf2_inverse(self.K1_bits)


# ============================================================
# LowMC-L3 round-key schedule
# ============================================================

class LowMCGrainRoundKey192(object):
    """
    v4-compatible round-key interface, but using Grain-generated LowMC K1.
    """

    def __init__(self, public_params):
        self.n = public_params.n
        self.K1 = public_params.K1_bits
        self.K1_inv = public_params.K1_inv_bits
        self.K1_int = public_params.K1_int
        self.K1_inv_int = public_params.K1_inv_int

    def round_key_1(self, master_key_bits):
        return mat_vec_mul_gf2(self.K1, master_key_bits)

    def recover_master_from_rk1(self, rk1_bits):
        return mat_vec_mul_gf2(self.K1_inv, rk1_bits)

    def round_key_1_int(self, master_key_int):
        return mat_vec_mul_int(self.K1_int, master_key_int)

    def recover_master_from_rk1_int(self, rk1_int):
        return mat_vec_mul_int(self.K1_inv_int, rk1_int)


# ============================================================
# Noisy threshold leakage model from v4
# ============================================================

class NoisyRK1Oracle(object):
    """
    Each rk1 bit b is embedded as an analog mean:

        +MU if b = 1
        -MU if b = 0

    Query:
        output = 1 iff mean + noise + probe >= 0

    This is a synthetic leakage oracle for research/self-study.
    """

    def __init__(self, rk1_bits, mu=MU, noise_bound=NOISE_BOUND):
        self.rk1 = rk1_bits[:]
        self.mu = float(mu)
        self.noise_bound = float(noise_bound)

    def sample_bit(self, idx, probe=0.0):
        mean = self.mu if self.rk1[idx] else -self.mu
        noise = random.uniform(
            -self.noise_bound,
            self.noise_bound
        )

        val = mean + noise + probe

        return 1 if val >= 0.0 else 0

    def sample_vector(self, probe=0.0):
        return [
            self.sample_bit(i, probe)
            for i in range(len(self.rk1))
        ]


# ============================================================
# Diagnostic pair generators from v4
# ============================================================

def pairgenerate_sym(
    oracle,
    idx,
    trials=PAIR_TRIALS,
    amplitude=None
):
    if amplitude is None:
        amplitude = 0.60 * oracle.noise_bound

    same = 0
    diff = 0

    for _ in range(trials):
        y1 = oracle.sample_bit(idx, probe=+amplitude)
        y2 = oracle.sample_bit(idx, probe=-amplitude)

        if y1 == y2:
            same += 1
        else:
            diff += 1

    return same, diff


def pairgenerate_change(
    oracle,
    idx,
    trials=PAIR_TRIALS,
    base_probe=0.0,
    eps=0.20
):
    same = 0
    diff = 0

    for _ in range(trials):
        y1 = oracle.sample_bit(idx, probe=base_probe)
        y2 = oracle.sample_bit(
            idx,
            probe=base_probe + eps
        )

        if y1 == y2:
            same += 1
        else:
            diff += 1

    return same, diff


# ============================================================
# Statistical rk1 recovery from v4
# ============================================================

def estimate_rk1_bit_majority(
    oracle,
    idx,
    trials=RK_RECOVERY_TRIALS,
    probe=0.0
):
    ones = 0

    for _ in range(trials):
        ones += oracle.sample_bit(
            idx,
            probe=probe
        )

    bit_hat = 1 if (2 * ones >= trials) else 0

    confidence = abs(
        (float(ones) / float(trials)) - 0.5
    )

    return bit_hat, ones, confidence


def recover_rk1(
    oracle,
    trials=RK_RECOVERY_TRIALS,
    verbose=True
):
    rk1_hat = [0] * NBITS
    confs = []

    for i in range(NBITS):
        bit_hat, ones, conf = estimate_rk1_bit_majority(
            oracle,
            i,
            trials=trials,
            probe=0.0
        )

        rk1_hat[i] = bit_hat
        confs.append(conf)

        if verbose:
            print(
                "rk1 bit {:03d}: "
                "ones={:3d}/{:3d} "
                "conf={:.4f} -> {}".format(
                    i,
                    ones,
                    trials,
                    conf,
                    bit_hat
                )
            )

    avg_conf = sum(confs) / float(len(confs))

    return rk1_hat, avg_conf


# ============================================================
# Consistency tests
# ============================================================

def verify_grain_K1_list_vs_int(lowmc, master_key_bits):
    """
    Verify that the converted v4-style MSB-list representation and the
    integer-row LowMC representation compute exactly the same RK1.
    """

    master_key_int = bitlist_msb_to_int(master_key_bits)

    rk1_list = lowmc.round_key_1(master_key_bits)
    rk1_int = lowmc.round_key_1_int(master_key_int)
    rk1_int_as_list = int_to_bitlist_msb(rk1_int, NBITS)

    return rk1_list == rk1_int_as_list


def verify_inverse_both_representations(
    lowmc,
    master_key_bits,
    rk1_bits
):
    master_from_list = lowmc.recover_master_from_rk1(
        rk1_bits
    )

    rk1_int = bitlist_msb_to_int(rk1_bits)

    master_from_int = lowmc.recover_master_from_rk1_int(
        rk1_int
    )

    master_from_int_bits = int_to_bitlist_msb(
        master_from_int,
        NBITS
    )

    return (
        master_from_list == master_key_bits
        and master_from_int_bits == master_key_bits
    )


# ============================================================
# End-to-end experiment
# ============================================================
# ============================================================
# End-to-end / 1000-random-key experiment
# ============================================================

NUM_EXPERIMENTS = 1000
PRINT_EVERY = 10


def run_single_trial(lowmc, trial_no=None):
    """
    Run one independent random-master-key leakage/key-recovery trial.
    LowMC public parameters (including K1) are fixed and generated once,
    because they are public deterministic LowMC parameters.
    """
    master_key = rand_bits(NBITS)
    rk1_true = lowmc.round_key_1(master_key)

    # Representation/inverse consistency checks
    same_operation = verify_grain_K1_list_vs_int(lowmc, master_key)
    inverse_ok = verify_inverse_both_representations(
        lowmc, master_key, rk1_true
    )
    if not same_operation:
        raise AssertionError("K1 multiplication representation mismatch.")
    if not inverse_ok:
        raise AssertionError("K1 inverse representation mismatch.")

    oracle = NoisyRK1Oracle(
        rk1_true,
        mu=MU,
        noise_bound=NOISE_BOUND
    )

    t0 = time.perf_counter()
    rk1_hat, avg_conf = recover_rk1(
        oracle,
        trials=RK_RECOVERY_TRIALS,
        verbose=False
    )
    master_hat = lowmc.recover_master_from_rk1(rk1_hat)
    elapsed = time.perf_counter() - t0

    rk1_hd = hamming_distance(rk1_hat, rk1_true)
    master_hd = hamming_distance(master_hat, master_key)

    return {
        "trial": trial_no,
        "master_key": master_key,
        "rk1_true": rk1_true,
        "rk1_hat": rk1_hat,
        "master_hat": master_hat,
        "rk1_match": (rk1_hd == 0),
        "master_match": (master_hd == 0),
        "rk1_hd": rk1_hd,
        "master_hd": master_hd,
        "avg_conf": avg_conf,
        "recovery_time": elapsed,
        "representation_match": same_operation,
        "inverse_match": inverse_ok,
    }


def run_1000_experiments(
    num_experiments=NUM_EXPERIMENTS,
    verbose_grain=True,
    print_every=PRINT_EVERY
):
    print("=" * 78)
    print("LowMC-L3 Grain-SSG noisy leakage: random-key repeated experiment")
    print("=" * 78)
    print("Experiments         =", num_experiments)
    print("n / k / m / r       = {} / {} / {} / {}".format(
        NBITS, NKEY, M_SBOX, ROUNDS
    ))
    print("MU                  =", MU)
    print("NOISE_BOUND         =", NOISE_BOUND)
    print("RK_RECOVERY_TRIALS  =", RK_RECOVERY_TRIALS)
    print("Random seed         =", RANDOM_SEED)
    print("=" * 78)

    # Public LowMC parameters are deterministic, so generate them once.
    print("\n[1] Generate deterministic LowMC-L3 public parameters once")
    t_pub0 = time.perf_counter()
    public_params = LowMCL3PublicParameters(verbose=verbose_grain)
    lowmc = LowMCGrainRoundKey192(public_params)
    t_pub = time.perf_counter() - t_pub0

    k1_rank = gf2_rank(public_params.K1_bits)
    print("rank(K1)                         =", k1_rank)
    print("public-parameter generation time = {:.3f} sec".format(t_pub))

    rk1_success = 0
    master_success = 0
    rk1_hd_sum = 0
    master_hd_sum = 0
    rk1_hd_max = 0
    master_hd_max = 0
    confidence_sum = 0.0
    recovery_time_sum = 0.0
    recovery_time_min = float("inf")
    recovery_time_max = 0.0

    failed_trials = []

    print("\n[2] Run {} random master-key trials".format(num_experiments))
    t_all0 = time.perf_counter()

    for trial in range(1, num_experiments + 1):
        result = run_single_trial(lowmc, trial_no=trial)

        rk1_success += int(result["rk1_match"])
        master_success += int(result["master_match"])

        rk1_hd_sum += result["rk1_hd"]
        master_hd_sum += result["master_hd"]
        rk1_hd_max = max(rk1_hd_max, result["rk1_hd"])
        master_hd_max = max(master_hd_max, result["master_hd"])

        confidence_sum += result["avg_conf"]
        recovery_time_sum += result["recovery_time"]
        recovery_time_min = min(
            recovery_time_min, result["recovery_time"]
        )
        recovery_time_max = max(
            recovery_time_max, result["recovery_time"]
        )

        if not result["master_match"]:
            failed_trials.append({
                "trial": trial,
                "rk1_hd": result["rk1_hd"],
                "master_hd": result["master_hd"],
                "master_key_hex": bits_to_hex(result["master_key"]),
                "master_hat_hex": bits_to_hex(result["master_hat"]),
            })

        if (
            trial == 1
            or trial % print_every == 0
            or trial == num_experiments
        ):
            print(
                "[{:4d}/{:4d}] rk1 success={:4d} ({:6.2f}%) | "
                "master success={:4d} ({:6.2f}%) | "
                "last HD(rk1/key)={:3d}/{:3d} | "
                "last time={:.4f}s".format(
                    trial,
                    num_experiments,
                    rk1_success,
                    100.0 * rk1_success / trial,
                    master_success,
                    100.0 * master_success / trial,
                    result["rk1_hd"],
                    result["master_hd"],
                    result["recovery_time"],
                )
            )

    total_time = time.perf_counter() - t_all0

    avg_rk1_hd = rk1_hd_sum / float(num_experiments)
    avg_master_hd = master_hd_sum / float(num_experiments)
    avg_confidence = confidence_sum / float(num_experiments)
    avg_recovery_time = recovery_time_sum / float(num_experiments)

    print("\n" + "=" * 78)
    print("[Final summary]")
    print("=" * 78)
    print(
        "RK1 exact recovery       = {}/{} = {:.2f}%".format(
            rk1_success,
            num_experiments,
            100.0 * rk1_success / num_experiments
        )
    )
    print(
        "Master-key recovery      = {}/{} = {:.2f}%".format(
            master_success,
            num_experiments,
            100.0 * master_success / num_experiments
        )
    )
    print(
        "Average RK1 HD           = {:.6f} bits".format(avg_rk1_hd)
    )
    print(
        "Maximum RK1 HD           = {} bits".format(rk1_hd_max)
    )
    print(
        "Average master-key HD    = {:.6f} bits".format(avg_master_hd)
    )
    print(
        "Maximum master-key HD    = {} bits".format(master_hd_max)
    )
    print(
        "Average confidence       = {:.6f}".format(avg_confidence)
    )
    print(
        "Average recovery time    = {:.6f} sec/trial".format(
            avg_recovery_time
        )
    )
    print(
        "Min/Max recovery time    = {:.6f} / {:.6f} sec".format(
            recovery_time_min, recovery_time_max
        )
    )
    print(
        "1000-trial loop time     = {:.3f} sec".format(total_time)
    )
    print(
        "Public parameter time    = {:.3f} sec (excluded from per-trial time)".format(
            t_pub
        )
    )
    print(
        "Failed master-key trials = {}".format(len(failed_trials))
    )

    if failed_trials:
        print("\nFirst up to 10 failed trials:")
        for f in failed_trials[:10]:
            print(
                " trial {:4d}: rk1_HD={:3d}, key_HD={:3d}, "
                "true={}, recovered={}".format(
                    f["trial"],
                    f["rk1_hd"],
                    f["master_hd"],
                    f["master_key_hex"],
                    f["master_hat_hex"],
                )
            )

    return {
        "num_experiments": num_experiments,
        "rk1_success": rk1_success,
        "master_success": master_success,
        "rk1_success_rate": rk1_success / float(num_experiments),
        "master_success_rate": master_success / float(num_experiments),
        "avg_rk1_hd": avg_rk1_hd,
        "max_rk1_hd": rk1_hd_max,
        "avg_master_hd": avg_master_hd,
        "max_master_hd": master_hd_max,
        "avg_confidence": avg_confidence,
        "avg_recovery_time": avg_recovery_time,
        "min_recovery_time": recovery_time_min,
        "max_recovery_time": recovery_time_max,
        "total_trial_time": total_time,
        "public_parameter_time": t_pub,
        "failed_trials": failed_trials,
        "K1_rank": k1_rank,
    }


def main():
    # Reproducible 1000-key experiment.
    # Change/remove this seed if a different random sequence is desired.
    random.seed(RANDOM_SEED)

    run_1000_experiments(
        num_experiments=NUM_EXPERIMENTS,
        verbose_grain=True,
        print_every=PRINT_EVERY
    )


if __name__ == "__main__":
    main()
