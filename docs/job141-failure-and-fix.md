# Job141 failure and tested fix

```json
{
  "time": "2026-09-02T22:45:26.371996Z",
  "job_id": 141,
  "failure_class": "experimental_configuration",
  "failure": "Custom validation adapter used a nonexistent Namespace field; no optimizer step verified.",
  "smallest_fix": "Use args.apply_chat_template_kwargs, matching native Miles session renderer configuration.",
  "sample_fields_changed": false,
  "runtime_validation": {
    "unchanged_native_samples": 32,
    "negative_controls": 7,
    "launcher_tests_passed": 28
  },
  "independent_metrics_fix": {
    "description": "Preserve existing spec_info counters across sample wire protocol; previous trainer-side zero MTP acceptance is unqualified.",
    "codec_tests_passed": 20,
    "proof": "tests/02-mtp-wire-candidate-test-v1"
  },
  "miles_sha": "df8cde5bedc0b19603b3f53433b5a00b76e1d687",
  "local_test_setup": "Initial broad pytest collection failed because minimal launcher environment lacks huggingface_hub. Existing isolated launch suite passed with confcutdir; no package changes.",
  "next_action": "Submit attempt9 after CPU evaluation build142 terminates and queue is empty."
}
```
