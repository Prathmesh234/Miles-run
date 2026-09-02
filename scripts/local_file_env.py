"""Bounded file-only task runtime behind OpenEnv; no host command execution.

Policy root filesystems are read-only. Only /app/task_file and temporary paths
are writable. A paused filesystem snapshot is graded in a fresh container after
the policy container is stopped. No grader output is returned to the policy.
This profile deliberately does not claim support for persistent-service tasks.
"""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import threading
import time
import uuid

from evidence import atomic, sha256, utcnow


MAX_ARCHIVE = 256 * 1024**2
MAX_OUTPUT = 16 * 1024**2


def atomic_bytes(path, payload):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_archive(payload, root_name):
    if len(payload) > MAX_ARCHIVE:
        raise ValueError('Task snapshot exceeds the bounded archive budget.')
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:') as archive:
        names, total = set(), 0
        for member in archive:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != root_name
                    or str(path) in names or not (member.isfile() or member.isdir())):
                raise ValueError('Snapshot contains a linked, special, duplicated or escaping entry.')
            names.add(str(path))
            total += member.size
            if total > MAX_ARCHIVE or len(names) > 10000:
                raise ValueError('Task snapshot exceeds its size/file-count budget.')
        if not names:
            raise ValueError('Empty task snapshot.')


def tree_archive(source, root_name, overrides=None):
    overrides = overrides or {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for path in [source] + sorted(source.rglob('*')):
            relative = path.relative_to(source)
            actual = overrides.get(str(relative), path)
            if path.is_symlink() or actual.is_symlink() or not (actual.is_file() or actual.is_dir()):
                raise ValueError('Linked/special staging source refused.')
            name = str(PurePosixPath(root_name) / str(relative))
            archive.add(actual, arcname=name, recursive=False)
            if buffer.tell() > MAX_ARCHIVE:
                raise ValueError('Staging archive exceeds its byte budget.')
    payload = buffer.getvalue()
    validate_archive(payload, root_name)
    return payload


def archive_contents(payload, root_name):
    """Upload *inside* a writable tmpfs, never through its read-only parent."""
    validate_archive(payload, root_name)
    output = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:') as source, tarfile.open(fileobj=output, mode='w') as target:
        for member in source:
            parts = PurePosixPath(member.name).parts
            if len(parts) == 1:
                continue
            member.name = str(PurePosixPath(*parts[1:]))
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
    return output.getvalue()


def container_options(image, name, run_id, role, mounts=None):
    if not image.startswith('sha256:') or len(image) != 71:
        raise ValueError('Task containers require an immutable local image ID.')
    return dict(image=image, name=name, entrypoint='/bin/sleep', command=['infinity'], detach=True, runtime='runc',
                working_dir='/app/task_file', network_mode='none', read_only=True,
                cap_drop=['ALL'], security_opt=['no-new-privileges:true'],
                mem_limit='1g', memswap_limit='1g', nano_cpus=1_000_000_000, pids_limit=128,
                ulimits=[{'Name': 'nofile', 'Soft': 1024, 'Hard': 1024}],
                mounts=mounts or [],
                tmpfs={'/tmp': 'rw,nosuid,nodev,size=128m,mode=1777', '/run': 'rw,nosuid,nodev,size=16m,mode=0755'},
                labels={'posttrainingx.run': run_id, 'posttrainingx.role': role},
                environment={'LANG': 'C.UTF-8', 'UV_OFFLINE': '1', 'UV_PYTHON_DOWNLOADS': 'never',
                             'UV_PYTHON_PREFERENCE': 'only-managed', 'PYTHONNOUSERSITE': '1'},
                log_config={'Type': 'json-file', 'Config': {'max-size': '1m', 'max-file': '1'}})


class FileTaskSession:
    def __init__(self, run_dir, manifest_path, task_id, *, purpose='policy', client=None):
        import docker
        if purpose not in ('policy', 'oracle', 'isolation-test'):
            raise ValueError('Unknown session purpose.')
        self.root = Path(run_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_relative_to(self.root):
            raise ValueError('Image manifest must be inside this run.')
        manifest = json.loads(self.manifest_path.read_text())
        matches = [row for row in manifest['images'] if row['task_id'] == task_id]
        if len(matches) != 1 or manifest['findings']:
            raise ValueError('Task is not in the qualified image manifest.')
        self.task = matches[0]
        self.task_id, self.purpose = task_id, purpose
        self.source = self.root / self.task['source_relpath']
        self.harness = self.root / self.task['runtime_harness_relpath']
        self.client = client or docker.from_env(timeout=20)
        self.episode = uuid.uuid4().hex
        self.directory = self.manifest_path.parent / 'episodes' / self.episode
        self.directory.mkdir(parents=True, mode=0o700)
        self.container = self.grader = None
        self.volumes = []
        self.sealed = False
        self.lock = threading.RLock()
        self.command_index = 0
        self.started = time.monotonic()
        self.record('created', image_manifest_sha256=sha256(self.manifest_path))
        try:
            self.guard()
            self.container = self.client.containers.run(**container_options(
                self.task['policy_image_id'], 'ptx-' + purpose + '-' + self.episode, self.root.name, purpose,
                self.workspace_mounts(purpose)))
            self.verify_container(self.container, purpose)
            initial = self.root / self.task['initial_tree_archive_relpath']
            if initial.is_symlink() or sha256(initial) != self.task['initial_tree_archive_sha256']:
                raise ValueError('Initialized public image filesystem changed.')
            data = initial.read_bytes()
            validate_archive(data, 'task_file')
            if not self.container.put_archive('/app/task_file', archive_contents(data, 'task_file')):
                raise RuntimeError('Public task upload failed.')
            self.record('ready', container_id=self.container.id, staged_public_sha256=hashlib.sha256(data).hexdigest())
        except Exception:
            self.close()
            raise

    def workspace_mounts(self, role):
        from docker.types import Mount
        targets = [('/app/task_file', '768m')]
        if role == 'grader':
            targets += [('/tests', '64m'), ('/logs/verifier', '64m'), ('/root/.cache/uv', '512m')]
        mounts = []
        for index, (target, size) in enumerate(targets):
            volume = self.client.volumes.create(name=f'ptx-{self.episode}-{role}-{index}', driver='local',
                driver_opts={'type': 'tmpfs', 'device': 'tmpfs', 'o': f'size={size},nosuid,nodev'},
                labels={'posttrainingx.run': self.root.name, 'posttrainingx.episode': self.episode})
            self.volumes.append(volume)
            mounts.append(Mount(target=target, source=volume.name, type='volume'))
            self.record('workspace_volume_created', volume=volume.name, target=target, size=size)
        return mounts

    def record(self, event, **fields):
        row = dict(time=utcnow(), monotonic_s=time.monotonic(), event=event,
                   episode_id=self.episode, task_id=self.task_id, purpose=self.purpose, **fields)
        with (self.directory / 'events.jsonl').open('a') as output:
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
            os.fsync(output.fileno())

    def guard(self):
        if shutil.disk_usage(self.root).free < 128 * 1024**3:
            raise RuntimeError('Run free-space reserve below 128 GiB.')
        if time.monotonic() - self.started > 900:
            raise TimeoutError('Environment lifetime exceeded 900 seconds.')

    def verify_container(self, container, role):
        container.reload()
        attrs = container.attrs
        config = attrs['HostConfig']
        if (attrs['Image'] != self.task['grader_image_id' if role == 'grader' else 'policy_image_id']
                or config['NetworkMode'] != 'none' or not config['ReadonlyRootfs']
                or config.get('Privileged') or config.get('Binds') or config.get('Devices')
                or set(config['CapDrop']) != {'ALL'} or config['PidsLimit'] != 128
                or config['Memory'] != 1024**3 or config['MemorySwap'] != 1024**3
                or config['NanoCpus'] != 1_000_000_000
                or attrs['Config']['Labels'].get('posttrainingx.run') != self.root.name):
            raise RuntimeError('Resolved container isolation/resources differ from the profile.')
        atomic(self.directory / (role + '-inspect.json'), attrs)

    def run_command(self, command, *, grading=False, timeout_s=60):
        with self.lock:
            if self.sealed and not grading:
                raise RuntimeError('Policy session is permanently sealed.')
            self.guard()
            container = self.grader if grading else self.container
            if container is None:
                raise RuntimeError('No active task container.')
            self.command_index += 1
            prefix = self.directory / f'{"grader" if grading else "policy"}-{self.command_index:04d}'
            args = ['/bin/bash', '--noprofile', '--norc', '-c', command]
            atomic(prefix.with_suffix('.command.json'), {'argv': args, 'container_id': container.id,
                                                       'timeout_s': timeout_s, 'purpose': self.purpose})
            execution = self.client.api.exec_create(container.id, args, workdir='/app/task_file', stdout=True, stderr=True)
            expired = threading.Event()

            def terminate():
                expired.set()
                try:
                    container.kill()
                except Exception as exc:
                    self.record('watchdog_error', error_type=type(exc).__name__)

            watchdog = threading.Timer(timeout_s, terminate)
            watchdog.daemon = True
            watchdog.start()
            total, started = 0, time.monotonic()
            stream = None
            try:
                with prefix.with_suffix('.out').open('xb') as out, prefix.with_suffix('.err').open('xb') as err:
                    stream = self.client.api.exec_start(execution['Id'], stream=True, demux=True)
                    for stdout, stderr in stream:
                        total += len(stdout or b'') + len(stderr or b'')
                        if expired.is_set() or total > MAX_OUTPUT:
                            terminate()
                            raise TimeoutError('Command timeout or output budget exceeded.')
                        out.write(stdout or b'')
                        err.write(stderr or b'')
                info = self.client.api.exec_inspect(execution['Id'])
                if expired.is_set() or info['Running'] or info['ExitCode'] is None:
                    raise TimeoutError('Command did not finish within the enforced budget.')
                self.record('command_finished', command_index=self.command_index, grading=grading,
                            duration_s=time.monotonic() - started, exit_code=info['ExitCode'], output_bytes=total)
                # Grader output is evidence only. It never enters an observation.
                output = '' if grading else (prefix.with_suffix('.out').read_text(errors='replace') +
                                               prefix.with_suffix('.err').read_text(errors='replace'))
                return info['ExitCode'], output
            except Exception:
                self.sealed = True
                terminate()
                self.record('command_failed', command_index=self.command_index, grading=grading,
                            duration_s=time.monotonic() - started)
                raise
            finally:
                watchdog.cancel()
                if stream is not None:
                    stream.close()

    def bounded_archive(self, container, path, cap=MAX_ARCHIVE):
        stream, _ = container.get_archive(path)
        output = bytearray()
        try:
            for chunk in stream:
                if len(output) + len(chunk) > cap:
                    raise ValueError('Container archive exceeds its byte budget.')
                output.extend(chunk)
        finally:
            stream.close()
        return bytes(output)

    def seal(self):
        if self.sealed:
            raise RuntimeError('Session was already sealed.')
        self.sealed = True
        self.container.pause()
        self.container.reload()
        if not self.container.attrs['State']['Paused']:
            raise RuntimeError('Policy processes did not freeze.')
        self.record('policy_paused')
        payload = self.bounded_archive(self.container, '/app/task_file')
        validate_archive(payload, 'task_file')
        atomic_bytes(self.directory / 'task-snapshot.tar', payload)
        self.container.unpause()
        self.container.kill()
        self.container.wait(timeout=20)
        self.container.reload()
        if self.container.attrs['State']['Running'] or self.container.attrs['State']['Pid'] != 0:
            raise RuntimeError('Policy processes survived sealing; no grader will start.')
        self.record('policy_stopped', snapshot_sha256=hashlib.sha256(payload).hexdigest())
        return payload

    def evaluate(self):
        with self.lock:
            try:
                self.guard()
                if sha256(self.harness) != self.task['offline_harness_sha256']:
                    raise ValueError('Pinned offline harness changed.')
                payload = self.seal()
                self.grader = self.client.containers.run(**container_options(
                    self.task['grader_image_id'], 'ptx-grader-' + self.episode, self.root.name, 'grader',
                    self.workspace_mounts('grader')))
                self.verify_container(self.grader, 'grader')
                tests = tree_archive(self.source / 'tests', 'tests', {'test.sh': self.harness})
                if (not self.grader.put_archive('/app/task_file', archive_contents(payload, 'task_file'))
                        or not self.grader.put_archive('/tests', archive_contents(tests, 'tests'))):
                    raise RuntimeError('Grader-only snapshot or tests upload failed.')
                self.record('grader_assets_staged', grader_container_id=self.grader.id,
                            tests_archive_sha256=hashlib.sha256(tests).hexdigest())
                self.run_command('/bin/bash --noprofile --norc /tests/test.sh', grading=True, timeout_s=300)
                verdict = self.bounded_archive(self.grader, '/logs/verifier/reward.txt', cap=1024*1024)
                with tarfile.open(fileobj=io.BytesIO(verdict), mode='r:') as archive:
                    members = archive.getmembers()
                    if len(members) != 1 or not members[0].isfile() or members[0].size > 64:
                        raise ValueError('Invalid verdict archive.')
                    text = archive.extractfile(members[0]).read().decode().strip()
                if text not in ('0', '1'):
                    raise ValueError('Canonical grader did not return a binary verdict.')
                result = {'reward': float(text), 'done': True, 'output': '', 'error': '',
                          'info': {'harness': 'tests/test.sh', 'offline_harness_sha256': self.task['offline_harness_sha256'],
                                   'grading_boundary': 'sealed-file-snapshot-v1'}}
                atomic(self.directory / 'result.json', result)
                self.record('graded', reward=result['reward'])
                return result
            except Exception as exc:
                self.sealed = True
                self.record('grading_error', error_type=type(exc).__name__)
                result = {'reward': None, 'done': True, 'output': '', 'error': 'grader_error',
                          'info': {'harness': 'tests/test.sh', 'grading_boundary': 'sealed-file-snapshot-v1'}}
                atomic(self.directory / 'result.json', result)
                # Preserve diagnostic details outside observations and training data.
                atomic(self.directory / 'grader-error.json', {'error_type': type(exc).__name__, 'message': str(exc)})
                return result
            finally:
                self.close()

    def close(self):
        with self.lock:
            self.sealed = True
            failures = []
            for container in (self.container, self.grader):
                if container is None:
                    continue
                try:
                    container.reload()
                    if container.attrs['Config']['Labels'].get('posttrainingx.run') != self.root.name:
                        raise ValueError('Refusing cleanup of a container not owned by this run.')
                    if container.attrs['State'].get('Paused'):
                        container.unpause()
                    container.remove(force=True)
                    self.record('container_removed', container_id=container.id)
                except Exception as exc:
                    if getattr(exc, 'status_code', None) != 404:
                        failures.append({'container_id': container.id, 'error_type': type(exc).__name__})
            self.container = self.grader = None
            for volume in self.volumes:
                try:
                    volume.reload()
                    if volume.attrs.get('Labels', {}).get('posttrainingx.episode') != self.episode:
                        raise ValueError('Refusing removal of another session volume.')
                    volume.remove()
                    self.record('workspace_volume_removed', volume=volume.name)
                except Exception as exc:
                    if getattr(exc, 'status_code', None) != 404:
                        failures.append({'volume': volume.name, 'error_type': type(exc).__name__})
            self.volumes = []
            self.client.close()
            if failures:
                atomic(self.directory / 'cleanup-errors.json', failures)
                raise RuntimeError('Run-owned container cleanup failed; see private evidence.')
