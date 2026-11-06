#!/usr/bin/env python3
"""LowMC-L5 v7: separate Boolean/ReLU equivalence and white-box RK1 activation recovery.

Threat model: attacker sees BOTH final ReLU post-activations for each coordinate
of the K1*master_key computation. This is explicit white-box instrumentation,
not an attack on standard Picnic or a realistic restricted side channel.
"""
import argparse
import csv
import importlib.util
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load ' + filename)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class LastGateTracer:
    """Capture only last XOR's two real ReLU activations per RK1 coordinate."""
    def __init__(self, supports):
        self.last = {i: 'row{}_xor{}'.format(i, len(s)-1)
                     for i, s in enumerate(supports) if len(s) >= 2}
        self.nodes = {}
        self.raw_events = 0
    def record_relu(self, round_index, scope, phase, box, coord, gate,
                    node, pre_activation, post_activation):
        if scope != 'key_schedule' or round_index != 1:
            return
        if gate != self.last.get(coord):
            return
        self.raw_events += 1
        key = (coord, node)
        if key in self.nodes:
            raise AssertionError('duplicate activation event: {}'.format(key))
        self.nodes[key] = (float(pre_activation), float(post_activation))

def recover_from_activations(v6, supports, master_key, capture_rows=False):
    """Evaluate the genuine ReLU K1 network, reconstruct RK1 from its last XOR nodes."""
    bits = v6.int_to_lsb_float_vector(master_key, len(supports))
    tracer = LastGateTracer(supports)
    # Deliberately discard return value: recovery only uses captured activation nodes.
    v6.relu_matvec(bits, supports, tracer, round_index=1,
                   scope='key_schedule', phase='K1_times_master_key')
    recovered = 0
    records = []
    direct_wires = 0
    for i, indices in enumerate(supports):
        if not indices:
            bit = 0  # a known zero row (not expected for full-rank K1)
        elif len(indices) == 1:
            # A one-input row has no ReLU node. This is explicitly a wire observation.
            direct_wires += 1
            bit = int(bits[indices[0]])
        else:
            a = tracer.nodes.get((i, 'relu_x_minus_y'))
            b = tracer.nodes.get((i, 'relu_y_minus_x'))
            if a is None or b is None:
                raise AssertionError('missing ReLU activation at RK1 coordinate {}'.format(i))
            value = a[1] + b[1]
            if abs(value - round(value)) > 1e-9 or round(value) not in (0, 1):
                raise AssertionError('nonbinary RK1 activation {}'.format(value))
            bit = int(round(value))
            if capture_rows:
                records.extend([
                    dict(coord=i, node='relu_x_minus_y', pre_activation=a[0], post_activation=a[1]),
                    dict(coord=i, node='relu_y_minus_x', pre_activation=b[0], post_activation=b[1]),
                ])
        recovered |= bit << i
    return recovered, tracer.raw_events, direct_wires, records

def write_csv(path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def equivalence(v5, v6, public, relu_public, count, seed, out, progress):
    rng = random.Random(seed)
    rows = []
    start = time.perf_counter()
    for i in range(1, count + 1):
        key = rng.getrandbits(256)
        plaintext = rng.getrandbits(256)
        t0 = time.perf_counter()
        ok = True
        error = ''
        cb = cr = None
        try:
            cb, tb = v6.encrypt_boolean_lowmc(v5, public, plaintext, key)
            cr, tr = v6.encrypt_relu_lowmc(public, relu_public, plaintext, key, tracer=None)
            v6.compare_traces(tb, tr, trial_no=i)
            ok = cb == cr
        except Exception as exc:
            ok = False
            error = str(exc)
        rows.append(dict(trial=i, pass_equivalence=int(ok), plaintext_hex=f'{plaintext:064x}',
                         master_key_hex=f'{key:064x}', boolean_ciphertext_hex=f'{cb:064x}' if cb is not None else '',
                         relu_ciphertext_hex=f'{cr:064x}' if cr is not None else '',
                         seconds=round(time.perf_counter()-t0, 6), error=error))
        if i == 1 or i % progress == 0 or i == count:
            print(f'[equivalence {count}] {i}/{count} pass={sum(x["pass_equivalence"] for x in rows)}', flush=True)
    write_csv(out / f'equivalence_{count}.csv', rows, list(rows[0]))
    return dict(campaign='equivalence', trials=count, success=sum(r['pass_equivalence'] for r in rows),
                seconds=round(time.perf_counter()-start, 4))

def recovery(v5, v6, public, relu_public, count, seed, out, progress):
    rng = random.Random(seed)
    supports = relu_public.key_supports[1]
    rows = []
    start = time.perf_counter()
    for i in range(1, count + 1):
        key = rng.getrandbits(256)
        t0 = time.perf_counter()
        rk_true = v5.mat_vec_mul_int(public.K1_int, key)
        rk_hat, events, wires, activation_rows = recover_from_activations(
            v6, supports, key, capture_rows=(i == 1))
        key_hat = v5.mat_vec_mul_int(public.K1_inv_int, rk_hat)
        rows.append(dict(trial=i, rk1_success=int(rk_hat == rk_true),
                         master_success=int(key_hat == key), rk1_hd=(rk_hat ^ rk_true).bit_count(),
                         master_hd=(key_hat ^ key).bit_count(),
                         true_key_hex=f'{key:064x}', recovered_key_hex=f'{key_hat:064x}',
                         true_rk1_hex=f'{rk_true:064x}', recovered_rk1_hex=f'{rk_hat:064x}',
                         captured_relu_nodes=events, direct_wire_coordinates=wires,
                         seconds=round(time.perf_counter()-t0, 6)))
        if i == 1:
            write_csv(out / f'activation_sample_{count}.csv', activation_rows,
                      ['coord', 'node', 'pre_activation', 'post_activation'])
        if i == 1 or i % progress == 0 or i == count:
            print(f'[recovery {count}] {i}/{count} master success={sum(x["master_success"] for x in rows)}', flush=True)
    write_csv(out / f'activation_recovery_{count}.csv', rows, list(rows[0]))
    return dict(campaign='activation_recovery', trials=count,
                rk1_success=sum(r['rk1_success'] for r in rows),
                success=sum(r['master_success'] for r in rows),
                max_master_hd=max(r['master_hd'] for r in rows),
                seconds=round(time.perf_counter()-start, 4),
                activation_nodes_per_trial=rows[0]['captured_relu_nodes'],
                direct_wires_per_trial=rows[0]['direct_wire_coordinates'])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all','equiv','recovery'], default='all')
    p.add_argument('--trials', type=int, nargs='+', choices=[10,1000], default=[10,1000])
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--progress', type=int, default=10)
    p.add_argument('--results', type=Path, default=ROOT / 'results')
    args = p.parse_args()
    if args.progress < 1:
        p.error('--progress must be positive')
    args.results.mkdir(parents=True, exist_ok=True)
    v5 = load('lowmc_v5', 'lowmc_L5_grain_keyrecovery_v5_1000.py')
    v6 = load('lowmc_v6', 'lowmc_L5_grain_relu_instrumented_v6.py')
    print('Generating Grain-SSG public parameters...', flush=True)
    public = v5.LowMCL5PublicParameters(verbose=False)
    relu_public = v6.ReLUPublicData(public)
    assert v5.gf2_rank_int(public.K1_int, 256) == 256
    assert v6.verify_sbox_equivalence()
    summary_path = args.results / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else []
    for n in args.trials:
        if args.mode in ('all', 'equiv'):
            result = equivalence(v5,v6,public,relu_public,n,args.seed+n,args.results,args.progress)
            summary = [x for x in summary if (x['campaign'],x['trials']) != (result['campaign'],result['trials'])]
            summary.append(result)
        if args.mode in ('all', 'recovery'):
            result = recovery(v5,v6,public,relu_public,n,args.seed+n+100000,args.results,args.progress)
            summary = [x for x in summary if (x['campaign'],x['trials']) != (result['campaign'],result['trials'])]
            summary.append(result)
        write_csv(args.results / 'summary.csv', summary,
                  ['campaign','trials','success','seconds','rk1_success','max_master_hd',
                   'activation_nodes_per_trial','direct_wires_per_trial'])
        (args.results / 'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Saved outputs to', args.results, flush=True)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
