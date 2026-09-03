"""Read-only serialized MXFP8 audit: hashes, tensor coverage, and BF16 exceptions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from convert_qwen_mxfp8 import PACKED, METADATA_FILES
from evidence import sha256


def expected_tensors(rows):
    expected = {}
    for row in rows:
        name, shape = row['name'], row['shape']
        packed = PACKED.fullmatch(name)
        if packed:
            prefix, kind = packed.groups()
            projections = ('gate', 'up') if kind == 'gate_up_proj' else ('down',)
            weights = [(f'{prefix}.{i}.{p}_proj.weight', [shape[1] // len(projections), shape[2]])
                       for i in range(shape[0]) for p in projections]
        else:
            weights = [(name, shape)]
        for key, dimensions in weights:
            if key in expected:
                raise ValueError('Duplicate expected output key.')
            quantized = row['precision'] == 'mxfp8'
            expected[key] = (dimensions, 'F8_E4M3' if quantized else row['dtype'])
            if quantized:
                expected[key.removesuffix('.weight') + '.weight_scale_inv'] = (
                    [*dimensions[:-1], dimensions[-1] // 32], 'U8')
    return expected


def headers(root, filenames):
    result = {}
    for filename in filenames:
        if Path(filename).name != filename:
            raise ValueError('Nonlocal checkpoint shard path.')
        with (root / filename).open('rb') as source:
            size = struct.unpack('<Q', source.read(8))[0]
            if size > 64 * 1024**2:
                raise ValueError('Header exceeds size guard.')
            data = json.loads(source.read(size))
        for key, item in data.items():
            if key == '__metadata__':
                continue
            if key in result:
                raise ValueError('Duplicate serialized tensor.')
            result[key] = dict(item, filename=filename, payload_start=8 + size)
    return result


def tensor_hash(root, header):
    start, end = header['data_offsets']
    remaining = end - start
    digest = hashlib.sha256()
    with (root / header['filename']).open('rb') as source:
        source.seek(header['payload_start'] + start)
        while remaining:
            chunk = source.read(min(remaining, 8 * 1024**2))
            if not chunk:
                raise ValueError('Truncated tensor payload.')
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def audit(source, candidate, source_manifest):
    marker = json.loads((candidate / 'CONVERSION_COMPLETE.json').read_text())
    if marker['checksums_sha256'] != sha256(candidate / 'checksums.sha256'):
        raise ValueError('Completion marker checksum differs.')
    checked = []
    for line in (candidate / 'checksums.sha256').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        if Path(name).name != name or (candidate / name).is_symlink() or sha256(candidate / name) != checksum:
            raise ValueError('Candidate checksum differs: ' + name)
        checked.append(name)
    plan = json.loads((candidate / 'plan.json').read_text())
    converted = json.loads((candidate / 'conversion.json').read_text())
    selected = {row['name'] for row in plan['tensors'] if row['precision'] == 'mxfp8'}
    if len(converted['metrics']) != len(selected) or {row['name'] for row in converted['metrics']} != selected:
        raise ValueError('Conversion numerical audit coverage differs.')
    if any(not math.isfinite(row['relative_l2']) or row['relative_l2'] > 0.06 for row in converted['metrics']):
        raise ValueError('Conversion numerical audit exceeded its bound.')
    frozen = {row['path']: row['sha256'] for row in source_manifest['files']['files']}
    if any(frozen[name] != checksum for name, checksum in converted['input_sha256'].items()):
        raise ValueError('Conversion source differs from the frozen downloaded model.')
    if len(converted['input_sha256']) != 26 or frozen['config.json'] != plan['config_sha256']:
        raise ValueError('Incomplete source/configuration provenance.')
    index = json.loads((candidate / 'model.safetensors.index.json').read_text())
    actual = headers(candidate, sorted(set(index['weight_map'].values())))
    expected = expected_tensors(plan['tensors'])
    if set(actual) != set(expected) or set(actual) != set(index['weight_map']):
        raise ValueError('Serialized tensor key coverage differs.')
    file_bytes = {name: (candidate / name).stat().st_size for name in set(index['weight_map'].values())}
    sizes = {'F8_E4M3': 1, 'U8': 1, 'BOOL': 1, 'BF16': 2, 'F16': 2, 'F32': 4, 'I32': 4, 'I64': 8}
    for key, row in actual.items():
        if (row['shape'], row['dtype']) != expected[key] or index['weight_map'][key] != row['filename']:
            raise ValueError('Serialized shape/dtype/shard mismatch: ' + key)
        begin, end = row['data_offsets']
        if begin < 0 or end - begin != math.prod(row['shape']) * sizes[row['dtype']] or end + row['payload_start'] > file_bytes[row['filename']]:
            raise ValueError('Serialized tensor byte extent invalid: ' + key)
    original = headers(source, sorted(converted['input_sha256']))
    unchanged = [row['name'] for row in plan['tensors'] if row['precision'] == 'source']
    for key in unchanged:
        if tensor_hash(source, original[key]) != tensor_hash(candidate, actual[key]):
            raise ValueError('High-precision exception changed bytes: ' + key)
    metadata = [name for name in METADATA_FILES if name != 'config.json' and (candidate / name).is_file()]
    for name in metadata:
        if sha256(candidate / name) != frozen[name]:
            raise ValueError('Tokenizer/metadata changed: ' + name)
    if not {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'} <= set(metadata):
        raise ValueError('Required tokenizer/renderer metadata missing.')
    payload = sum(row['data_offsets'][1] - row['data_offsets'][0] for row in actual.values())
    if payload != index['metadata']['total_size'] or payload != converted['payload_bytes']:
        raise ValueError('Serialized payload accounting differs.')
    return dict(status='serialized_candidate_validated_not_runtime_qualified', tensor_count=len(actual),
                payload_bytes=payload, source_payload_bytes=plan['source_payload_bytes'],
                packed_expert_tensors=plan['packed_experts'], unchanged_tensors_byte_exact=len(unchanged),
                frozen_input_shards=26, checksummed_files=len(checked), unchanged_metadata=metadata,
                quantized_source_tensors=len(converted['metrics']),
                max_relative_l2=max(row['relative_l2'] for row in converted['metrics']),
                checksums_sha256=marker['checksums_sha256'], optimizer_steps_enabled=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.candidate, json.loads(args.source_manifest.read_text())), indent=2))
