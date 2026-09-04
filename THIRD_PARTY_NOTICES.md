# Third-party provenance

The experiment uses Miles at `70b89e11770fc9bac984e22cfff89c51cca44203` and SGLang at `c16b821ef3177a688a073c173b44c0ce48b5bf3e`. Full upstream repositories and model weights are not redistributed here.

The SGLang Qwen router source snapshots, which were adapted upstream from vLLM, remain in the raw archives and the prior publication commit `0fa4636863b1f61b444a830f74980cb06d59c10e`. Their original copyright/SPDX notices remain intact. A copy of their Apache 2.0 license is retained in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

The current [router patch installer](install_precision_patch.py) and [precision validation](sglang_precision.py) describe the job-local FP32 router calculation with unchanged BF16 weight storage. [ppo-resident-broadcast.patch](ppo-resident-broadcast.patch) is a separate Miles lifecycle fix with its source commit and attribution recorded in the patch header. Exact before/after hashes remain in each run archive.

Only compact task-level measurements are included in the current publication. This repository does not claim ownership of upstream models, task content, framework source, or their licenses.
