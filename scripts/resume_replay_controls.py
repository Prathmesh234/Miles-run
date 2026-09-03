"""Opt-in numerical controls and read-only optimizer-shard attribution.

The control keeps the original checkpoint comparison strict. Changing the
execution profile does not retroactively qualify the nondeterministic run.
"""

DETERMINISTIC_ENV = {
    'NCCL_ALGO': 'Ring',
    'NVTE_ALLOW_NONDETERMINISTIC_ALGO': '0',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
}


def apply_control(args, profile, environ):
    if profile not in ('original', 'deterministic'):
        raise ValueError('Unknown replay execution profile: ' + profile)
    original = args.deterministic_mode
    if profile == 'deterministic':
        if any(environ.get(key) != value for key, value in DETERMINISTIC_ENV.items()):
            raise ValueError('Deterministic control environment differs from the frozen profile.')
        args.deterministic_mode = True
    return dict(profile=profile, original_deterministic_mode=original,
                resolved_deterministic_mode=args.deterministic_mode,
                environment={key: environ.get(key) for key in DETERMINISTIC_ENV},
                comparison_tolerance=0,
                scope='Numerical replay control only; original checkpoint equality remains mandatory.')


def tensor_identity(tensor):
    return str(tensor.device), tensor.data_ptr(), tensor.numel(), str(tensor.dtype)


def optimizer_owners(models, optimizer):
    """Associate native state views with model names without copying tensors."""
    names = {}
    for chunk, model in enumerate(models):
        for name, param in model.named_parameters():
            names.setdefault(id(param), []).append(f'model[{chunk}].{name}')
    owners = {}
    for chain, opt in enumerate(getattr(optimizer, 'chained_optimizers', [optimizer])):
        for buffer, dtype_maps in enumerate(opt.gbuf_ranges):
            for dtype, buckets in dtype_maps.items():
                for bucket, mapping in enumerate(buckets):
                    for param, ranges in mapping['param_map'].items():
                        if id(param) not in names:
                            raise ValueError('Optimizer parameter is absent from the live model.')
                        for state_name, tensor in opt._get_main_param_and_optimizer_states(param).items():
                            # Step tensors can be shared across parameters; they are not
                            # part of the distributed parameter/moment buffer.
                            if state_name not in ('param', 'exp_avg', 'exp_avg_sq'):
                                continue
                            key = tensor_identity(tensor)
                            owner = dict(model_parameters=names[id(param)], model_shape=list(param.shape),
                                optimizer_index=chain, buffer_index=buffer, bucket_index=bucket,
                                buffer_dtype=str(dtype), state_name=state_name,
                                ranges={name: [value.start, value.end] for name, value in ranges.items()})
                            if key in owners and owners[key] != owner:
                                raise ValueError('Optimizer tensor view has ambiguous ownership.')
                            owners[key] = owner
    return owners


def annotate_mismatches(rows, state, owners):
    """Attach native shard coordinates and live owners only to failed leaves."""
    failed = {row['path']: row for row in rows if not row['equal'] and row.get('required_for_resume', True)}
    if not failed:
        return

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + '/' + str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path + '/' + str(index))
        elif path in failed and hasattr(value, 'global_offset') and hasattr(value, 'data'):
            row = failed[path]
            row['native_shard'] = dict(key=value.key, global_offset=list(value.global_offset),
                                       local_shape=list(value.local_shape))
            if path.startswith('state/optimizer/'):
                row['optimizer_owner'] = owners.get(tensor_identity(value.data))
                row['owner_resolution'] = 'matched_live_view' if row['optimizer_owner'] else 'unresolved'

    visit(state, 'state')
