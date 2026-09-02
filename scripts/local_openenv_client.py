"""Small OpenEnv WS client; no OpenEnv/OpenAI dependency upgrade in GPU image.

Carries only environment actions/observations. Policy sampling, renderer, token
IDs and logprobs stay in the existing Miles session server and agent loop.
"""
import asyncio
import json
import re
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit


class Action(SimpleNamespace):
    pass


class LocalOpenEnvClient:
    def __init__(self, base_url, message_timeout_s=360):
        url = urlsplit(base_url)
        if url.scheme not in ('http', 'https') or url.username or url.password or url.query or url.fragment:
            raise ValueError('Expected a plain configured environment endpoint.')
        self.url = urlunsplit(('wss' if url.scheme == 'https' else 'ws', url.netloc, url.path.rstrip('/') + '/ws', '', ''))
        self.timeout = message_timeout_s
        self.ws = None
        self.episode_id = None
        self.task_id = None

    async def __aenter__(self):
        from websockets.asyncio.client import connect
        self.ws = await connect(self.url, proxy=None, open_timeout=20, max_size=20*1024**2)
        return self

    async def __aexit__(self, *exc):
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({'type': 'close'}))
            finally:
                await self.ws.close()

    async def _request(self, operation, data):
        await self.ws.send(json.dumps({'type': operation, 'data': data}))
        reply = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=self.timeout))
        if reply.get('type') == 'error':
            error = reply.get('data', {})
            raise RuntimeError('Environment error: ' + str(error.get('code', 'UNKNOWN')))
        expected = 'state' if operation == 'state' else 'observation'
        if reply.get('type') != expected or not isinstance(reply.get('data'), dict):
            raise RuntimeError('Unexpected OpenEnv reply type.')
        return reply['data']

    async def exchange(self, operation, data):
        value = await self._request(operation, data)
        observation = SimpleNamespace(**value.get('observation', {}))
        return SimpleNamespace(observation=observation, reward=value.get('reward'), done=value.get('done'),
                               metadata=value.get('metadata'))

    async def reset(self, **kwargs):
        self.episode_id = self.task_id = None
        result = await self.exchange('reset', kwargs)
        state = await self._request('state', {})
        episode_id = state.get('episode_id')
        task_id = state.get('task_id')
        expected_task = kwargs.get('task_id') or kwargs.get('task_name')
        if (not isinstance(episode_id, str) or not re.fullmatch('[0-9a-f]{32}', episode_id)
                or not task_id or task_id != getattr(result.observation, 'task_id', None)
                or (expected_task and task_id != expected_task)):
            raise RuntimeError('OpenEnv episode/task identity did not reconcile after reset.')
        self.episode_id, self.task_id = episode_id, task_id
        return result

    async def step(self, action):
        return await self.exchange('step', vars(action))
