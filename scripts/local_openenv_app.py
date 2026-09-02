"""Maintained OpenEnv server protocol over a run-owned, sealed file-task backend."""
import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from tbench2_env.models import Tbench2Action, Tbench2Observation, Tbench2State

from local_file_env import FileTaskSession


class LocalFileEnvironment(Environment[Tbench2Action, Tbench2Observation, Tbench2State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.session = None
        self._state = Tbench2State()

    def reset(self, seed=None, episode_id=None, **kwargs):
        del seed, episode_id
        self.close()
        if kwargs.get('task_path') or kwargs.get('path'):
            raise ValueError('Only manifest-allowlisted task IDs are accepted.')
        task_id = kwargs.get('task_id') or kwargs.get('task_name')
        self.session = FileTaskSession(os.environ['PTX_RUN_DIR'], os.environ['PTX_TASK_IMAGES'], task_id)
        self._state = Tbench2State(episode_id=self.session.episode, task_id=task_id, terminal_ready=True)
        return Tbench2Observation(instruction=(self.session.source / 'instruction.md').read_text(),
                                 task_id=task_id, action_type='reset', output='', reward=0.0, done=False)

    def step(self, action, timeout_s=None, **kwargs):
        del timeout_s, kwargs
        if self.session is None:
            raise RuntimeError('No task session; reset first.')
        self._state.step_count += 1
        if action.action_type == 'exec':
            code, output = self.session.run_command(action.command)
            return Tbench2Observation(task_id=self.session.task_id, output=output,
                                     success=code == 0, action_type='exec', reward=None, done=False,
                                     info={'exit_code': code})
        if action.action_type == 'evaluate':
            result = self.session.evaluate()
            self._state.terminal_ready = False
            return Tbench2Observation(task_id=self.session.task_id, action_type='evaluate',
                                     success=not result['error'], **result)
        if action.action_type == 'close':
            self.close()
            return Tbench2Observation(action_type='close', done=True)
        raise ValueError('Unsupported action; file runtime allows exec/evaluate/close only.')

    @property
    def state(self):
        return self._state

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self._state.terminal_ready = False


app = create_app(LocalFileEnvironment, Tbench2Action, Tbench2Observation,
                 env_name='posttrainingx-local-file', max_concurrent_envs=int(os.getenv('MAX_CONCURRENT_ENVS', '4')))
