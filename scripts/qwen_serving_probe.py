"""Bounded EP8 serving validation inside the pinned, network-isolated GPU image."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

from evidence import Run, atomic, metric, sha256, utcnow


def attempt_suffix(attempt):
    return '' if attempt == 1 else '-attempt-' + str(attempt)


def case_name(mtp, attempt):
    return ('02-qwen-serving-mtp-on' if mtp else '02-qwen-serving-mtp-off') + attempt_suffix(attempt)


def server_command(model, mtp, precision='bf16'):
    if precision not in ('bf16', 'mxfp8'):
        raise ValueError('Unqualified serving precision.')
    command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(model),
        '--host', '127.0.0.1', '--port', '31872', '--tp-size', '8', '--ep-size', '8',
        '--nnodes', '1', '--node-rank', '0', '--context-length', '2048', '--max-total-tokens', '4096',
        '--max-running-requests', '2', '--mem-fraction-static', '0.7', '--cuda-graph-bs', '1', '2',
        '--dtype', 'bfloat16', '--random-seed', '1234',
        '--skip-server-warmup', '--enable-draft-weights-cpu-backup', '--enable-metrics',
        '--decode-log-interval', '1']
    if precision == 'mxfp8':
        command += ['--moe-runner-backend', 'flashinfer_trtllm_routed',
                    '--fp8-gemm-backend', 'flashinfer_trtllm']
    if mtp:
        command += ['--speculative-algorithm', 'EAGLE', '--speculative-num-steps', '2',
                    '--speculative-eagle-topk', '1', '--speculative-num-draft-tokens', '3',
                    '--mamba-scheduler-strategy', 'extra_buffer']
    return command


def request(path, payload=None, timeout=5):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request('http://127.0.0.1:31872' + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def prometheus_rows(text):
    from prometheus_client.parser import text_string_to_metric_families
    rows = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if not math.isfinite(sample.value):
                # At this pinned revision the timer explicitly invalidates this
                # optional gauge on startup/idle. Retain the unavailable sample;
                # do not invent a number or exempt other non-finite metrics.
                if sample.name == 'sglang:fwd_occupancy' and math.isnan(sample.value):
                    rows.append({'metric': 'collector_error', 'value': None, 'unit': 'event',
                                 'requested_metric': sample.name, 'labels': sample.labels,
                                 'error': 'Forward timing window unavailable',
                                 'reason': 'upstream_timer_window_unavailable', 'fatal': False})
                    continue
                rows.append({'metric': 'collector_error', 'value': None, 'unit': 'event',
                             'requested_metric': sample.name, 'labels': sample.labels,
                             'error': 'Non-finite Prometheus sample', 'fatal': True})
            else:
                rows.append({'metric': sample.name, 'value': sample.value, 'unit': 'exporter_native',
                             'labels': sample.labels})
    if not rows:
        raise ValueError('SGLang returned an empty metrics document.')
    return rows


def collect_metrics(root, stopped, errors):
    with (root / 'sglang.jsonl.partial').open('x') as normalized, (root / 'sglang.prometheus.jsonl.partial').open('x') as raw:
        while not stopped.is_set():
            tick = time.monotonic()
            common = {'time': utcnow(), 'monotonic_s': tick, 'hostname': socket.gethostname(),
                      'source': 'sglang-prometheus', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                      'role': 'rollout-serving-validation'}
            try:
                text = request('/metrics')
                raw.write(json.dumps(dict(common, text=text)) + '\n')
                rows = prometheus_rows(text)
            except Exception as exc:
                errors.append(str(exc))
                rows = [{'metric': 'collector_error', 'value': None, 'unit': 'event', 'error': str(exc)}]
            for row in rows:
                if row['metric'] == 'collector_error' and row.get('fatal', True) and row.get('error') not in errors:
                    errors.append(row.get('error', 'Metrics collector error'))
                normalized.write(json.dumps(dict(common, **row), allow_nan=False) + '\n')
            normalized.flush()
            raw.flush()
            stopped.wait(max(0, 1 - (time.monotonic()-tick)))
    for name in ('sglang.jsonl', 'sglang.prometheus.jsonl'):
        os.rename(root / (name + '.partial'), root / name)


def wait_ready(process, root):
    started = time.monotonic()
    with (root / 'startup.jsonl').open('x') as log:
        while time.monotonic() - started < 540:
            if process.poll() is not None:
                raise RuntimeError('SGLang exited before readiness: ' + str(process.returncode))
            try:
                request('/health', timeout=2)
                return time.monotonic() - started
            except (OSError, urllib.error.URLError) as exc:
                log.write(json.dumps({'time': utcnow(), 'monotonic_s': time.monotonic(),
                                     'state': 'starting', 'health_error': str(exc)}) + '\n')
                log.flush()
            time.sleep(1)
    raise TimeoutError('SGLang did not become ready within 540 seconds.')


def prompt_token_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                       add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('Chat renderer must return one nonempty flat list of integer token IDs.')
    return ids


def generate(root, tokenizer, index, prompt):
    ids = prompt_token_ids(tokenizer, prompt)
    payload = {'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 64, 'top_p': 1.0},
               'return_logprob': True, 'logprob_start_len': 0, 'stream': True}
    atomic(root / f'request-{index}.json', payload)
    req = urllib.request.Request('http://127.0.0.1:31872/generate', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first_text, final, events = None, None, []
    with urllib.request.urlopen(req, timeout=180) as response, (root / f'response-{index}.sse.jsonl').open('x') as log:
        for line in response:
            text = line.decode().strip()
            if not text.startswith('data:'):
                continue
            value = text[5:].strip()
            if value == '[DONE]':
                break
            data = json.loads(value)
            now = time.monotonic()
            event = {'time': utcnow(), 'monotonic_s': now, 'elapsed_s': now-start, 'data': data}
            log.write(json.dumps(event) + '\n')
            log.flush()
            events.append(now)
            if data.get('text') and first_text is None:
                first_text = now-start
            final = data
    if not final or not final.get('text') or final.get('meta_info', {}).get('completion_tokens', 0) <= 0:
        raise ValueError('SGLang did not return valid nonempty text with token accounting.')
    result = {'prompt': prompt, 'input_ids': ids, 'response': final, 'duration_s': time.monotonic()-start,
              'client_first_text_s': first_text, 'sse_chunk_interarrival_s': [b-a for a, b in zip(events, events[1:])],
              'note': 'SSE chunks can contain multiple speculative tokens; chunk arrival is not exact per-token ITL.'}
    atomic(root / f'generation-{index}.json', result)
    return result


def stop_owned_server(server, grace_s=30):
    """Signal only the parent so its watchdog can stop before workers exit."""
    import psutil
    result = {'requested_signal': None, 'forced_cleanup': False, 'errors': []}
    try:
        owned = psutil.Process(server.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        owned = []
    result['descendant_pids_before'] = [p.pid for p in owned]
    if server.poll() is None:
        server.terminate()
        result['requested_signal'] = 'SIGTERM_parent_only'
        try:
            server.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=10)
            result['forced_cleanup'] = True
            result['errors'].append('Owned SGLang process group exceeded graceful shutdown timeout.')
    _, alive = psutil.wait_procs(owned, timeout=5)
    live = []
    for process in alive:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                live.append(process)
        except psutil.NoSuchProcess:
            pass
    alive = live
    result['descendant_pids_after_grace'] = [p.pid for p in alive]
    if alive:
        result['forced_cleanup'] = True
        result['errors'].append('Owned SGLang descendants survived parent shutdown.')
        for process in alive:
            try:
                process.kill()  # psutil guards against PID reuse.
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(alive, timeout=5)
        result['descendant_pids_after_forced_cleanup'] = [p.pid for p in remaining]
    result['server_exit_code'] = server.returncode
    # This pinned SGLang kills its own process tree after graceful drain. Do not
    # call -9 a clean exit; require its explicit drain log when interpreting it.
    return result


def child(run, mtp, attempt):
    root = run.root / 'tests' / case_name(mtp, attempt)
    model = Path('/model')
    quantization = json.loads((model / 'config.json').read_text()).get('quantization_config', {})
    precision = quantization.get('quant_method', 'bf16')
    command = server_command(model, mtp, precision)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               SGLANG_ENABLE_METRICS_DEVICE_TIMER='1')
    stopped, metric_errors = threading.Event(), []
    collector, server = None, None
    result = {'started_at': utcnow(), 'mtp_enabled': mtp, 'precision': precision, 'command': command, 'errors': []}
    atomic(root / 'server-command.json', result)
    with (root / 'logs/server.out').open('x') as out, (root / 'logs/server.err').open('x') as err:
        try:
            server = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
            result['server_pid'] = server.pid
            result['startup_s'] = wait_ready(server, root)
            atomic(root / 'server-info.json', json.loads(request('/server_info', timeout=10)))
            collector = threading.Thread(target=collect_metrics, args=(root, stopped, metric_errors), daemon=True)
            collector.start()
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
            result['tokenizer_sha256'] = sha256(model / 'tokenizer.json')
            result['chat_template_sha256'] = sha256(model / 'chat_template.jinja')
            prompts = ['Return the integer result of 2 + 2, and nothing else.', 'Write exactly: infrastructure-ready']
            result['generations'] = [generate(root, tokenizer, i, prompt) for i, prompt in enumerate(prompts)]
            metrics = request('/metrics', timeout=10)
            atomic(root / 'metrics-after.prom', metrics)
            acceptance = [r for r in prometheus_rows(metrics) if 'accept' in r['metric'].lower() and 'spec' in r['metric'].lower()]
            result['mtp_acceptance_metrics'] = acceptance
            if mtp and not acceptance:
                raise ValueError('MTP enabled but no acceptance metric was exposed; preserve logs and stop.')
            if server.poll() is not None:
                raise RuntimeError('SGLang exited during validation.')
        except Exception as exc:
            result['errors'].append(str(exc))
            atomic(root / 'logs/exception.txt', traceback.format_exc())
        finally:
            stopped.set()
            if collector:
                collector.join(timeout=10)
                if collector.is_alive():
                    result['errors'].append('Metrics collector did not stop.')
            result['errors'].extend(metric_errors)
            if (root / 'sglang.jsonl').exists():
                rows = [json.loads(line) for line in (root / 'sglang.jsonl').read_text().splitlines()]
                result['metrics_coverage'] = {
                    'records': len(rows),
                    'unavailable_forward_timer_samples': sum(r.get('reason') == 'upstream_timer_window_unavailable' for r in rows),
                    'finite_forward_timer_samples': sum(r['metric'] == 'sglang:fwd_occupancy' for r in rows),
                    'scope': 'Smoke exporter coverage only; required full-workload metrics gate remains separate.'}
            if server is not None:
                result['cleanup'] = stop_owned_server(server)
                result['errors'].extend(result['cleanup']['errors'])
                result['server_exit_code'] = server.returncode
                logs = (root / 'logs/server.err').read_text() + (root / 'logs/server.out').read_text()
                drained = 'Gracefully exiting... Remaining number of requests 0.' in logs
                crashed = 'crashed with exit code' in logs or 'SIGQUIT received' in logs
                result['cleanup']['upstream_zero_request_drain_logged'] = drained
                if crashed or (server.returncode not in (0, -signal.SIGTERM) and not drained):
                    result['errors'].append('Server shutdown lacks a clean exit or verified zero-request drain.')
    result['ended_at'] = utcnow()
    atomic(root / 'probe-result.json', result)
    print(json.dumps({'mtp_enabled': mtp, 'errors': result['errors'], 'startup_s': result.get('startup_s')}), flush=True)
    return int(bool(result['errors']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--mtp', action='store_true')
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    if args.child:
        return child(run, args.mtp, args.attempt)
    for mtp in (False, True):
        phase = run.phase(case_name(mtp, args.attempt))
        rc, _, _ = phase.command([sys.executable, str(Path(__file__).resolve()), '--run-dir', str(run.root),
                                 '--child', '--attempt', str(args.attempt)] + (['--mtp'] if mtp else []), timeout=950)
        result_file = phase.path / 'probe-result.json'
        data = json.loads(result_file.read_text()) if result_file.exists() else {'errors': ['Probe result missing.']}
        results = [metric('server_startup', data['startup_s'], 's', socket.gethostname())] if 'startup_s' in data else []
        for index, generation in enumerate(data.get('generations', [])):
            results += [metric('generation_duration', generation['duration_s'], 's', request_index=index),
                        metric('client_first_text', generation['client_first_text_s'], 's', request_index=index)]
        phase.finish('fail' if rc else 'ok', results=results, exit_code=rc,
            failure_summary='; '.join(data['errors']) or 'Serving child failed.' if rc else None,
            metadata={'mtp_enabled': mtp, 'findings': data['errors'], 'server_exit_code': data.get('server_exit_code'),
                      'scope': 'Single-node EP8 serving validation; not RL or a throughput benchmark.',
                      'artifacts': [str(phase.path.relative_to(run.root))]}, refresh=False)
        if rc:
            if not mtp:
                run.phase(case_name(True, args.attempt)).finish('skip', reason='mtp_off_serving_gate_failed', refresh=False)
            return rc
    return 0


if __name__ == '__main__':
    sys.exit(main())
