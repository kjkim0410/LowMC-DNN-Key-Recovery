#!/usr/bin/env python3
"""
lowmc_L3_grain_relu_instrumented_v6.py

Extension of:

    lowmc_L3_grain_keyrecovery_v5_1000.py

Goals
-----
1. Reuse exactly the same Grain-SSG LowMC-L3 public parameters:
       L_1,...,L_20
       C_1,...,C_20
       K_0,...,K_20

2. Add a complete Boolean LowMC-L3 encryption path.

3. Add an exactly equivalent real-valued / ReLU LowMC-L3 path
   for binary inputs.

4. Verify Boolean-vs-ReLU equivalence:
       - round keys
       - initial whitening
       - S-box layer
       - linear layer
       - constant addition
       - round-key addition
       - ciphertext

5. Instrument ReLU pre-activations and post-activations.

6. Optionally export activation traces to CSV.

Important
---------
This is an implementation / research instrumentation experiment.

The ReLU realization is an exact realization only on the Boolean domain
{0,1}.  The activation tracer is explicit white-box instrumentation.
It does NOT imply that standard Picnic signatures expose these values.

No NumPy / PyTorch required.
Python 3.8+.
"""

import argparse
import csv
import importlib.util
import random
import time

from dataclasses import dataclass, asdict
from pathlib import Path


# =====================================================================
# Base program
# =====================================================================

DEFAULT_BASE_FILE = "lowmc_L3_grain_keyrecovery_v5_1000.py"


def load_base_module(filename):
    """
    Dynamically import the supplied v5_1000 program.

    Parentheses in the filename prevent an ordinary 'import ...' statement,
    so importlib is used.
    """

    candidates = [
        Path(filename),
        Path(__file__).resolve().parent / filename,
    ]

    path = None

    for candidate in candidates:
        if candidate.is_file():
            path = candidate.resolve()
            break

    if path is None:
        raise FileNotFoundError(
            "Cannot locate base file: {}\n"
            "Place it in the same directory as this program or use --base."
            .format(filename)
        )

    spec = importlib.util.spec_from_file_location(
        "lowmc_l3_v5_base",
        str(path)
    )

    if spec is None or spec.loader is None:
        raise ImportError("Cannot create import specification for {}".format(path))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module, path


# =====================================================================
# Boolean LowMC-L3
# =====================================================================

def lowmc_sbox3_bool(x):
    """
    Standard LowMC 3-bit S-box.

    Input:
        x = (a,b,c)
        a = bit 2
        b = bit 1
        c = bit 0

    Boolean equations:

        a' = a XOR (b AND c)

        b' = a XOR b XOR (a AND c)

        c' = a XOR b XOR c XOR (a AND b)
    """

    a = (x >> 2) & 1
    b = (x >> 1) & 1
    c = x & 1

    ap = a ^ (b & c)
    bp = a ^ b ^ (a & c)
    cp = a ^ b ^ c ^ (a & b)

    return (ap << 2) | (bp << 1) | cp


def bool_sbox_layer(state, n, m):
    """
    Apply m LowMC S-boxes to the least-significant 3*m bits.
    """

    out = state

    for box in range(m):
        shift = 3 * box

        x = (state >> shift) & 0x7
        y = lowmc_sbox3_bool(x)

        out &= ~(0x7 << shift)
        out |= y << shift

    return out & ((1 << n) - 1)


def bool_round_keys(base, public, master_key):
    """
    RK_i = K_i * master_key over GF(2).
    """

    return [
        base.mat_vec_mul_int(Ki, master_key)
        for Ki in public.round_key_matrices
    ]


def encrypt_boolean_lowmc(base, public, plaintext, master_key):
    """
    Complete Boolean LowMC-L3 encryption.

        state = plaintext XOR RK_0

        for i = 1,...,20:
            state = SboxLayer(state)
            state = L_i * state
            state = state XOR C_i
            state = state XOR RK_i
    """

    n = public.n
    mask_n = (1 << n) - 1

    plaintext &= mask_n
    master_key &= mask_n

    round_keys = bool_round_keys(
        base,
        public,
        master_key
    )

    state = plaintext ^ round_keys[0]

    trace = {
        "round_keys": round_keys[:],
        "after_initial_key": state,
        "rounds": [],
    }

    for r in range(1, public.r + 1):

        x0 = state

        x1 = bool_sbox_layer(
            x0,
            public.n,
            public.m
        )

        x2 = base.mat_vec_mul_int(
            public.linear_matrices[r - 1],
            x1
        )

        x3 = x2 ^ public.round_constants[r - 1]

        rk = round_keys[r]

        state = (x3 ^ rk) & mask_n

        trace["rounds"].append({
            "round": r,
            "input": x0,
            "after_sbox": x1,
            "after_linear": x2,
            "after_constant": x3,
            "round_key": rk,
            "output": state,
        })

    trace["ciphertext"] = state

    return state, trace


# =====================================================================
# Bit-vector conversion
# =====================================================================

def int_to_lsb_float_vector(x, n):
    """
    Integer -> LSB-first real-valued vector.

        v[i] = bit i
    """

    return [
        float((x >> i) & 1)
        for i in range(n)
    ]


def float_vector_to_int_lsb(v, tol=1.0e-9):
    """
    Convert an expected {0,1}-valued real vector back to integer.

    Raises an exception if a node deviates from the Boolean domain.
    """

    x = 0

    for i, value in enumerate(v):

        if abs(value) <= tol:
            bit = 0

        elif abs(value - 1.0) <= tol:
            bit = 1

        else:
            raise AssertionError(
                "Non-Boolean real value at coordinate {}: {!r}"
                .format(i, value)
            )

        x |= bit << i

    return x


def assert_binary_vector(v, label, tol=1.0e-9):

    for i, x in enumerate(v):

        if not (
            abs(x) <= tol
            or abs(x - 1.0) <= tol
        ):
            raise AssertionError(
                "{} contains a non-Boolean value at {}: {}"
                .format(label, i, x)
            )


# =====================================================================
# Activation instrumentation
# =====================================================================

@dataclass
class ActivationEvent:

    seq: int

    round_index: int

    scope: str
    phase: str

    box: int
    coord: int

    gate: str
    node: str

    pre_activation: float
    post_activation: float


class ActivationTracer(object):
    """
    Bounded ReLU activation recorder.

    scopes examples:

        sbox
        linear
        key_schedule
        state_xor

    Setting scopes=None records every scope.

    Setting rounds=None records all rounds.
    """

    def __init__(
        self,
        enabled=True,
        scopes=None,
        rounds=None,
        max_events=5000
    ):

        self.enabled = bool(enabled)

        self.scopes = (
            None if scopes is None
            else set(scopes)
        )

        self.rounds = (
            None if rounds is None
            else set(rounds)
        )

        self.max_events = int(max_events)

        self.events = []

        self.total_matching_events = 0
        self.truncated = False

    def _selected(self, scope, round_index):

        if not self.enabled:
            return False

        if (
            self.scopes is not None
            and scope not in self.scopes
        ):
            return False

        if (
            self.rounds is not None
            and round_index not in self.rounds
        ):
            return False

        return True

    def record_relu(
        self,
        round_index,
        scope,
        phase,
        box,
        coord,
        gate,
        node,
        pre_activation,
        post_activation
    ):

        if not self._selected(
            scope,
            round_index
        ):
            return

        self.total_matching_events += 1

        if len(self.events) >= self.max_events:
            self.truncated = True
            return

        event = ActivationEvent(

            seq=len(self.events),

            round_index=int(round_index),

            scope=str(scope),
            phase=str(phase),

            box=-1 if box is None else int(box),
            coord=-1 if coord is None else int(coord),

            gate=str(gate),
            node=str(node),

            pre_activation=float(pre_activation),
            post_activation=float(post_activation),
        )

        self.events.append(event)

    def write_csv(self, filename):

        if not self.events:
            print(
                "[trace] no events to write to {}".format(filename)
            )
            return

        fieldnames = list(
            asdict(self.events[0]).keys()
        )

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as fp:

            writer = csv.DictWriter(
                fp,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for event in self.events:
                writer.writerow(
                    asdict(event)
                )

        print(
            "[trace] wrote {} activation events to {}"
            .format(
                len(self.events),
                filename
            )
        )


# =====================================================================
# Exact ReLU Boolean gates
# =====================================================================

def relu_value(
    z,
    tracer=None,
    *,
    round_index=0,
    scope="generic",
    phase="generic",
    box=None,
    coord=None,
    gate="relu",
    node="relu"
):

    z = float(z)

    y = z if z > 0.0 else 0.0

    if tracer is not None:

        tracer.record_relu(

            round_index=round_index,

            scope=scope,
            phase=phase,

            box=box,
            coord=coord,

            gate=gate,
            node=node,

            pre_activation=z,
            post_activation=y,
        )

    return y


def relu_xor2(
    x,
    y,
    tracer=None,
    *,
    round_index=0,
    scope="generic",
    phase="xor",
    box=None,
    coord=None,
    gate="xor"
):
    """
    Exact XOR for x,y in {0,1}:

        XOR(x,y)
          = ReLU(x-y)
          + ReLU(y-x)
    """

    z_pos = float(x) - float(y)

    a_pos = relu_value(
        z_pos,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=coord,
        gate=gate,
        node="relu_x_minus_y"
    )

    z_neg = float(y) - float(x)

    a_neg = relu_value(
        z_neg,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=coord,
        gate=gate,
        node="relu_y_minus_x"
    )

    return a_pos + a_neg


def relu_and2(
    x,
    y,
    tracer=None,
    *,
    round_index=0,
    scope="generic",
    phase="and",
    box=None,
    coord=None,
    gate="and"
):
    """
    Exact AND for x,y in {0,1}:

        AND(x,y) = ReLU(x+y-1)
    """

    z = float(x) + float(y) - 1.0

    return relu_value(
        z,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=coord,
        gate=gate,
        node="relu_x_plus_y_minus_1"
    )


# =====================================================================
# ReLU LowMC 3-bit S-box
# =====================================================================

def lowmc_sbox3_relu(
    a,
    b,
    c,
    tracer=None,
    round_index=0,
    box=None
):
    """
    Exact real-valued/ReLU realization of the LowMC 3-bit S-box.

        a' = a XOR (b AND c)
        b' = a XOR b XOR (a AND c)
        c' = a XOR b XOR c XOR (a AND b)
    """

    scope = "sbox"
    phase = "sbox"

    # -------------------------------------------------
    # a' = a XOR (b AND c)
    # -------------------------------------------------

    bc = relu_and2(
        b,
        c,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="bc_and"
    )

    ap = relu_xor2(
        a,
        bc,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="ap_xor"
    )

    # -------------------------------------------------
    # b' = a XOR b XOR (a AND c)
    # -------------------------------------------------

    ac = relu_and2(
        a,
        c,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="ac_and"
    )

    ab_xor = relu_xor2(
        a,
        b,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="bp_xor_ab"
    )

    bp = relu_xor2(
        ab_xor,
        ac,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="bp_xor_ac"
    )

    # -------------------------------------------------
    # c' = a XOR b XOR c XOR (a AND b)
    # -------------------------------------------------

    ab = relu_and2(
        a,
        b,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="ab_and"
    )

    cp1 = relu_xor2(
        a,
        b,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="cp_xor_ab"
    )

    cp2 = relu_xor2(
        cp1,
        c,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="cp_xor_c"
    )

    cp = relu_xor2(
        cp2,
        ab,
        tracer,
        round_index=round_index,
        scope=scope,
        phase=phase,
        box=box,
        coord=None,
        gate="cp_xor_abterm"
    )

    return ap, bp, cp


def relu_sbox_layer(
    state,
    n,
    m,
    tracer=None,
    round_index=0
):
    """
    LSB-first state convention.

    Integer LowMC S-box input:

        bit 2 = a
        bit 1 = b
        bit 0 = c

    Therefore, for box j:

        state[3j+0] = c
        state[3j+1] = b
        state[3j+2] = a
    """

    out = state[:]

    for box in range(m):

        base = 3 * box

        c = state[base + 0]
        b = state[base + 1]
        a = state[base + 2]

        ap, bp, cp = lowmc_sbox3_relu(
            a,
            b,
            c,
            tracer=tracer,
            round_index=round_index,
            box=box
        )

        out[base + 0] = cp
        out[base + 1] = bp
        out[base + 2] = ap

    assert_binary_vector(
        out,
        "ReLU S-box output, round {}".format(round_index)
    )

    return out


# =====================================================================
# ReLU GF(2) matrix multiplication
# =====================================================================

def matrix_row_supports(rows, ncols):
    """
    Precompute positions of 1 bits in every binary matrix row.
    """

    return [
        [
            j
            for j in range(ncols)
            if (row >> j) & 1
        ]
        for row in rows
    ]


def relu_parity_from_indices(
    vector,
    indices,
    tracer=None,
    *,
    round_index,
    scope,
    phase,
    coord
):
    """
    Compute XOR of selected binary coordinates using chained
    exact ReLU XOR gates.
    """

    if not indices:
        return 0.0

    acc = vector[indices[0]]

    for step, j in enumerate(indices[1:], start=1):

        acc = relu_xor2(
            acc,
            vector[j],
            tracer,
            round_index=round_index,
            scope=scope,
            phase=phase,
            box=None,
            coord=coord,
            gate="row{}_xor{}".format(
                coord,
                step
            )
        )

    return acc


def relu_matvec(
    vector,
    supports,
    tracer=None,
    *,
    round_index,
    scope,
    phase
):

    out = []

    for row_index, indices in enumerate(supports):

        y = relu_parity_from_indices(
            vector,
            indices,
            tracer,
            round_index=round_index,
            scope=scope,
            phase=phase,
            coord=row_index
        )

        out.append(y)

    assert_binary_vector(
        out,
        "{} round {}".format(
            phase,
            round_index
        )
    )

    return out


# =====================================================================
# Vector XOR
# =====================================================================

def relu_vector_xor(
    x,
    y,
    tracer=None,
    *,
    round_index,
    phase
):

    if len(x) != len(y):
        raise ValueError("Vector sizes differ")

    out = []

    for i, (a, b) in enumerate(zip(x, y)):

        out.append(
            relu_xor2(
                a,
                b,
                tracer,
                round_index=round_index,
                scope="state_xor",
                phase=phase,
                box=None,
                coord=i,
                gate="coord{}_xor".format(i)
            )
        )

    assert_binary_vector(
        out,
        "{} round {}".format(
            phase,
            round_index
        )
    )

    return out


# =====================================================================
# ReLU public representation
# =====================================================================

class ReLUPublicData(object):

    def __init__(self, public):

        self.n = public.n
        self.k = public.k
        self.m = public.m
        self.r = public.r

        self.linear_supports = [
            matrix_row_supports(
                Li,
                public.n
            )
            for Li in public.linear_matrices
        ]

        self.key_supports = [
            matrix_row_supports(
                Ki,
                public.k
            )
            for Ki in public.round_key_matrices
        ]

        self.constants = [
            int_to_lsb_float_vector(
                ci,
                public.n
            )
            for ci in public.round_constants
        ]


# =====================================================================
# ReLU key schedule
# =====================================================================

def relu_generate_round_keys(
    relu_public,
    master_key_vector,
    tracer=None
):

    round_keys = []

    for i, supports in enumerate(
        relu_public.key_supports
    ):

        rk = relu_matvec(
            master_key_vector,
            supports,
            tracer,
            round_index=i,
            scope="key_schedule",
            phase="K{}_times_master_key".format(i)
        )

        round_keys.append(rk)

    return round_keys


# =====================================================================
# Complete ReLU LowMC encryption
# =====================================================================

def encrypt_relu_lowmc(
    public,
    relu_public,
    plaintext,
    master_key,
    tracer=None
):
    """
    Complete real-valued/ReLU LowMC-L3 encryption.

    Every Boolean XOR is implemented as

        ReLU(x-y) + ReLU(y-x)

    Every Boolean AND is implemented as

        ReLU(x+y-1)

    All internal state values are checked to remain exactly in {0,1}.
    """

    n = public.n
    mask_n = (1 << n) - 1

    plaintext &= mask_n
    master_key &= mask_n

    plaintext_vector = int_to_lsb_float_vector(
        plaintext,
        n
    )

    master_key_vector = int_to_lsb_float_vector(
        master_key,
        n
    )

    # -------------------------------------------------
    # ReLU key schedule
    # -------------------------------------------------

    round_keys = relu_generate_round_keys(
        relu_public,
        master_key_vector,
        tracer=tracer
    )

    round_key_ints = [
        float_vector_to_int_lsb(rk)
        for rk in round_keys
    ]

    # -------------------------------------------------
    # Initial whitening
    # -------------------------------------------------

    state = relu_vector_xor(
        plaintext_vector,
        round_keys[0],
        tracer,
        round_index=0,
        phase="initial_key_xor"
    )

    trace = {
        "round_keys": round_key_ints,
        "after_initial_key":
            float_vector_to_int_lsb(state),
        "rounds": [],
    }

    # -------------------------------------------------
    # 20 LowMC rounds
    # -------------------------------------------------

    for r in range(1, public.r + 1):

        x0_int = float_vector_to_int_lsb(
            state
        )

        # ---------------------------------------------
        # nonlinear S-box layer
        # ---------------------------------------------

        x1 = relu_sbox_layer(
            state,
            public.n,
            public.m,
            tracer=tracer,
            round_index=r
        )

        x1_int = float_vector_to_int_lsb(
            x1
        )

        # ---------------------------------------------
        # linear layer
        # ---------------------------------------------

        x2 = relu_matvec(
            x1,
            relu_public.linear_supports[r - 1],
            tracer,
            round_index=r,
            scope="linear",
            phase="L{}_times_state".format(r)
        )

        x2_int = float_vector_to_int_lsb(
            x2
        )

        # ---------------------------------------------
        # round constant
        # ---------------------------------------------

        x3 = relu_vector_xor(
            x2,
            relu_public.constants[r - 1],
            tracer,
            round_index=r,
            phase="constant_xor"
        )

        x3_int = float_vector_to_int_lsb(
            x3
        )

        # ---------------------------------------------
        # round key
        # ---------------------------------------------

        state = relu_vector_xor(
            x3,
            round_keys[r],
            tracer,
            round_index=r,
            phase="round_key_xor"
        )

        state_int = float_vector_to_int_lsb(
            state
        )

        trace["rounds"].append({

            "round": r,

            "input": x0_int,

            "after_sbox": x1_int,

            "after_linear": x2_int,

            "after_constant": x3_int,

            "round_key":
                round_key_ints[r],

            "output": state_int,
        })

    ciphertext = float_vector_to_int_lsb(
        state
    )

    trace["ciphertext"] = ciphertext

    return ciphertext, trace


# =====================================================================
# Boolean-vs-ReLU consistency checks
# =====================================================================

def compare_traces(
    bool_trace,
    relu_trace,
    trial_no=None,
    hex_width=None
):

    prefix = (
        "trial {}".format(trial_no)
        if trial_no is not None
        else "trace"
    )

    if hex_width is None:
        # Infer a safe width from the ciphertext when no parameter object is passed.
        max_value = max(
            int(bool_trace.get("ciphertext", 0)),
            int(relu_trace.get("ciphertext", 0)),
            1
        )
        hex_width = max(1, (max_value.bit_length() + 3) // 4)

    if (
        bool_trace["round_keys"]
        != relu_trace["round_keys"]
    ):
        raise AssertionError(
            "{}: round-key mismatch"
            .format(prefix)
        )

    if (
        bool_trace["after_initial_key"]
        != relu_trace["after_initial_key"]
    ):
        raise AssertionError(
            "{}: initial whitening mismatch"
            .format(prefix)
        )

    for i, (b, r) in enumerate(
        zip(
            bool_trace["rounds"],
            relu_trace["rounds"]
        ),
        start=1
    ):

        for field in [
            "input",
            "after_sbox",
            "after_linear",
            "after_constant",
            "round_key",
            "output",
        ]:

            if b[field] != r[field]:

                raise AssertionError(
                    "{}: round {} field {} mismatch\n"
                    "Boolean = {:{w}x}\n"
                    "ReLU    = {:{w}x}"
                    .format(
                        prefix,
                        i,
                        field,
                        b[field],
                        r[field],
                        w=hex_width
                    )
                )

    if (
        bool_trace["ciphertext"]
        != relu_trace["ciphertext"]
    ):
        raise AssertionError(
            "{}: final ciphertext mismatch"
            .format(prefix)
        )

    return True


def verify_sbox_equivalence():

    for x in range(8):

        expected = lowmc_sbox3_bool(x)

        a = float((x >> 2) & 1)
        b = float((x >> 1) & 1)
        c = float(x & 1)

        ap, bp, cp = lowmc_sbox3_relu(
            a,
            b,
            c,
            tracer=None
        )

        got = (
            (int(round(ap)) << 2)
            | (int(round(bp)) << 1)
            | int(round(cp))
        )

        if got != expected:

            raise AssertionError(
                "S-box mismatch for x={}: "
                "Boolean={} ReLU={}"
                .format(
                    x,
                    expected,
                    got
                )
            )

    return True


def run_equivalence_campaign(
    base,
    public,
    relu_public,
    trials,
    seed
):

    rng = random.Random(seed)

    print("")
    print("=" * 78)
    print("[Boolean vs. real-valued/ReLU LowMC-L3 equivalence]")
    print("=" * 78)
    print("Trials =", trials)
    print("Seed   =", seed)

    t0 = time.perf_counter()

    for trial in range(
        1,
        trials + 1
    ):

        master_key = rng.getrandbits(
            public.k
        )

        plaintext = rng.getrandbits(
            public.n
        )

        c_bool, tr_bool = encrypt_boolean_lowmc(
            base,
            public,
            plaintext,
            master_key
        )

        c_relu, tr_relu = encrypt_relu_lowmc(
            public,
            relu_public,
            plaintext,
            master_key,
            tracer=None
        )

        compare_traces(
            tr_bool,
            tr_relu,
            trial_no=trial,
            hex_width=(public.n + 3) // 4
        )

        if c_bool != c_relu:
            raise AssertionError(
                "Ciphertext mismatch"
            )

        if (
            trial == 1
            or trial == trials
            or trial % 10 == 0
        ):

            print(
                "[{:4d}/{:4d}] PASS  ciphertext={:{w}x}"
                .format(
                    trial,
                    trials,
                    c_bool,
                    w=(public.n + 3) // 4
                )
            )

    elapsed = time.perf_counter() - t0

    print("")
    print(
        "Boolean/ReLU exact equivalence: PASS"
    )

    print(
        "Elapsed: {:.3f} sec".format(
            elapsed
        )
    )

    return True


# =====================================================================
# Activation trace demonstration
# =====================================================================

def print_activation_events(
    tracer,
    max_print=80
):

    print("")
    print("=" * 78)
    print("[ReLU activation instrumentation]")
    print("=" * 78)

    print(
        "stored events         =",
        len(tracer.events)
    )

    print(
        "matching events total =",
        tracer.total_matching_events
    )

    print(
        "truncated             =",
        tracer.truncated
    )

    print("")

    print(
        "{:>5s} {:>3s} {:>12s} {:>18s} "
        "{:>4s} {:>5s} {:>22s} {:>22s} "
        "{:>9s} {:>9s}".format(
            "seq",
            "rnd",
            "scope",
            "phase",
            "box",
            "coord",
            "gate",
            "node",
            "pre",
            "post"
        )
    )

    print("-" * 140)

    for event in tracer.events[:max_print]:

        print(
            "{:5d} {:3d} {:>12s} {:>18s} "
            "{:4d} {:5d} {:>22s} {:>22s} "
            "{:9.4f} {:9.4f}".format(

                event.seq,
                event.round_index,

                event.scope,
                event.phase,

                event.box,
                event.coord,

                event.gate,
                event.node,

                event.pre_activation,
                event.post_activation,
            )
        )


def run_activation_demo(
    base,
    public,
    relu_public,
    seed,
    scopes,
    trace_round,
    max_events,
    max_print,
    csv_filename=None
):

    rng = random.Random(
        seed
    )

    master_key = rng.getrandbits(
        public.k
    )

    plaintext = rng.getrandbits(
        public.n
    )

    if trace_round < 0:
        rounds = None
    else:
        rounds = {trace_round}

    tracer = ActivationTracer(
        enabled=True,
        scopes=scopes,
        rounds=rounds,
        max_events=max_events
    )

    c_bool, tr_bool = encrypt_boolean_lowmc(
        base,
        public,
        plaintext,
        master_key
    )

    c_relu, tr_relu = encrypt_relu_lowmc(
        public,
        relu_public,
        plaintext,
        master_key,
        tracer=tracer
    )

    compare_traces(
        tr_bool,
        tr_relu,
        hex_width=(public.n + 3) // 4
    )

    print("")
    print("=" * 78)
    print("[Instrumented encryption]")
    print("=" * 78)

    print(
        "Master key : {:{w}x}".format(
            master_key,
            w=(public.k + 3) // 4
        )
    )

    print(
        "Plaintext  : {:{w}x}".format(
            plaintext,
            w=(public.n + 3) // 4
        )
    )

    print(
        "Boolean CT : {:{w}x}".format(
            c_bool,
            w=(public.n + 3) // 4
        )
    )

    print(
        "ReLU CT    : {:{w}x}".format(
            c_relu,
            w=(public.n + 3) // 4
        )
    )

    print(
        "Equivalence: {}".format(
            "PASS"
            if c_bool == c_relu
            else "FAIL"
        )
    )

    print_activation_events(
        tracer,
        max_print=max_print
    )

    if csv_filename:

        tracer.write_csv(
            csv_filename
        )

    return tracer


# =====================================================================
# Scope parser
# =====================================================================

VALID_SCOPES = {
    "sbox",
    "linear",
    "key_schedule",
    "state_xor",
}


def parse_scopes(text):

    text = text.strip()

    if text.lower() == "all":
        return None

    scopes = {
        x.strip()
        for x in text.split(",")
        if x.strip()
    }

    unknown = scopes - VALID_SCOPES

    if unknown:

        raise ValueError(
            "Unknown trace scope(s): {}\n"
            "Valid: {}"
            .format(
                sorted(unknown),
                sorted(VALID_SCOPES)
            )
        )

    return scopes


# =====================================================================
# LowMC-L3 public-parameter constructor compatibility
# =====================================================================

def build_public_parameters(base, verbose=False):
    """
    Construct the LowMC-L3 public parameters from the companion
    lowmc_L3_grain_keyrecovery_v5_1000.py module.

    The preferred API is LowMCL3PublicParameters(verbose=...).
    A small fallback is provided for closely related base-file variants.
    """
    preferred = getattr(base, "LowMCL3PublicParameters", None)
    if preferred is not None:
        return preferred(verbose=verbose)

    generic = getattr(base, "LowMCPublicParameters", None)
    if generic is not None:
        try:
            return generic(verbose=verbose)
        except TypeError:
            return generic()

    candidates = [
        obj for name, obj in vars(base).items()
        if isinstance(obj, type)
        and "PublicParameters" in name
        and "L3" in name
    ]
    if len(candidates) == 1:
        try:
            return candidates[0](verbose=verbose)
        except TypeError:
            return candidates[0]()

    raise AttributeError(
        "Cannot find a LowMC-L3 public-parameter class in the base module. "
        "Expected LowMCL3PublicParameters."
    )


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Boolean-vs-ReLU LowMC-L3 equivalence "
            "and activation instrumentation"
        )
    )

    parser.add_argument(
        "--base",
        default=DEFAULT_BASE_FILE,
        help=(
            "Path to lowmc_L3_grain_keyrecovery_v5_1000.py"
        )
    )

    parser.add_argument(
        "--equiv-trials",
        type=int,
        default=5,
        help=(
            "Number of random Boolean-vs-ReLU equivalence trials "
            "(default: 5)"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260818,
        help="Random seed for equivalence tests"
    )

    parser.add_argument(
        "--trace-scopes",
        default="sbox",
        help=(
            "Comma-separated scopes: "
            "sbox,linear,key_schedule,state_xor "
            "or 'all'"
        )
    )

    parser.add_argument(
        "--trace-round",
        type=int,
        default=1,
        help=(
            "Round to instrument. "
            "-1 means every round. "
            "Use 0 for whitening/K0."
        )
    )

    parser.add_argument(
        "--trace-max-events",
        type=int,
        default=5000,
        help="Maximum stored activation events"
    )

    parser.add_argument(
        "--trace-print",
        type=int,
        default=80,
        help="Maximum activation rows printed"
    )

    parser.add_argument(
        "--csv",
        default="",
        help=(
            "Optional CSV output file for activation traces"
        )
    )

    parser.add_argument(
        "--skip-trace-demo",
        action="store_true",
        help="Skip the instrumented example encryption"
    )

    parser.add_argument(
        "--verbose-grain",
        action="store_true",
        help="Print Grain public-parameter generation progress"
    )

    parser.add_argument(
        "--run-original-1000",
        action="store_true",
        help=(
            "After equivalence tests, invoke the original "
            "1000-key noisy RK1 recovery experiment"
        )
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Load original v5_1000 program
    # -------------------------------------------------------------

    base, base_path = load_base_module(
        args.base
    )

    print("=" * 78)
    print("LowMC-L3 Boolean / real-valued ReLU extension")
    print("=" * 78)

    print(
        "Base program =",
        base_path
    )

    print(
        "n / k / m / r = {} / {} / {} / {}"
        .format(
            base.NBITS,
            base.NKEY,
            base.M_SBOX,
            base.ROUNDS
        )
    )

    # -------------------------------------------------------------
    # Validate LowMC S-box
    # -------------------------------------------------------------

    verify_sbox_equivalence()

    print(
        "Boolean/ReLU LowMC 3-bit S-box: PASS"
    )

    # -------------------------------------------------------------
    # Build exact same deterministic public parameters
    # -------------------------------------------------------------

    print("")
    print(
        "[1] Generate deterministic Grain-SSG LowMC-L3 parameters"
    )

    t0 = time.perf_counter()

    public = build_public_parameters(
        base,
        verbose=args.verbose_grain
    )

    elapsed_public = (
        time.perf_counter() - t0
    )

    print(
        "rank(K1) =",
        base.gf2_rank_int(
            public.K1_int,
            public.k
        )
    )

    print(
        "public parameter time = {:.3f} sec"
        .format(
            elapsed_public
        )
    )

    # -------------------------------------------------------------
    # Build ReLU matrix-support representation
    # -------------------------------------------------------------

    print("")
    print(
        "[2] Build ReLU-compatible public representation"
    )

    t1 = time.perf_counter()

    relu_public = ReLUPublicData(
        public
    )

    print(
        "ReLU public conversion time = {:.3f} sec"
        .format(
            time.perf_counter() - t1
        )
    )

    # -------------------------------------------------------------
    # Equivalence campaign
    # -------------------------------------------------------------

    if args.equiv_trials > 0:

        run_equivalence_campaign(
            base,
            public,
            relu_public,
            trials=args.equiv_trials,
            seed=args.seed
        )

    # -------------------------------------------------------------
    # Activation trace
    # -------------------------------------------------------------

    if not args.skip_trace_demo:

        scopes = parse_scopes(
            args.trace_scopes
        )

        run_activation_demo(
            base,
            public,
            relu_public,
            seed=args.seed + 1,
            scopes=scopes,
            trace_round=args.trace_round,
            max_events=args.trace_max_events,
            max_print=args.trace_print,
            csv_filename=(
                args.csv
                if args.csv
                else None
            )
        )

    # -------------------------------------------------------------
    # Optional original 1000-key recovery experiment
    # -------------------------------------------------------------

    if args.run_original_1000:

        print("")
        print("=" * 78)
        print("[Original v5_1000 RK1/master-key recovery experiment]")
        print("=" * 78)

        base.random.seed(
            base.RANDOM_SEED
        )

        base.run_1000_experiments(
            num_experiments=base.NUM_EXPERIMENTS,
            verbose_grain=True,
            print_every=base.PRINT_EVERY
        )


# if __name__ == "__main__":

#     base, base_path = load_base_module(
#         "lowmc_L3_grain_keyrecovery_v5_1000.py"
#     )

#     base.random.seed(
#         base.RANDOM_SEED
#     )

#     base.run_1000_experiments(
#         num_experiments=1000,
#         verbose_grain=True,
#         print_every=10
#     )

if __name__ == "__main__":
    main()