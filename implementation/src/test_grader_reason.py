"""An `unverified` verdict must say WHY the re-measurement failed.

`measure` already classifies the cause into `failure_reason`; the grader used to
discard it and log a bare "remeasured=0", which is indistinguishable between a wedged
box, a missing cache file, a compiler crash and a genuinely zero-throughput model.

That cost two full trn2.48xlarge runs. The 35B completed every stage, reported 368
tok/s of box throughput, and then failed grading with `remeasured=0` SEVENTEEN SECONDS
after Stage 6 -- far too fast to have loaded a 72 GB model, which is exactly what the
reason would have said.
"""

from __future__ import annotations

from trusted_grader import verify_winner


class _M:
    def __init__(self, metric, failure_reason="", top1=None):
        self.metric = metric
        self.failure_reason = failure_reason
        self.top1_tokens = top1 or []


class _Backend:
    def __init__(self, m):
        self._m = m

    def build_baseline(self, model_id):
        return object()

    def apply_config(self, artifact, cfg):
        return artifact

    def compile(self, artifact):
        return artifact

    def measure(self, neff, shape, batch):
        return self._m


class _Spec:
    model_id = "Qwen/Qwen3.5-35B-A3B"
    probe_shape = "2048/512"
    probe_batch = 1


class _Winner:
    metric = 6.33
    config = {"tp_degree": 8}


def test_a_failed_remeasure_reports_the_backend_reason():
    reason = ("device busy: cores held by another process (orphaned ranks from an "
              "earlier candidate?)")
    logs: list[str] = []
    res = verify_winner(_Backend(_M(0.0, reason)), _Spec(), _Winner(), [1, 2, 3],
                        logs.append)
    assert res["verdict"] == "unverified"
    assert res["remeasure_failure"] == reason
    assert any(reason in m for m in logs), logs


def test_the_log_line_still_carries_the_numbers():
    logs: list[str] = []
    verify_winner(_Backend(_M(0.0, "boom")), _Spec(), _Winner(), [1], logs.append)
    line = logs[0]
    assert "claimed=6" in line and "remeasured=0" in line
    assert "re-measure failed: boom" in line


def test_a_zero_with_no_reason_is_called_out_rather_than_left_blank():
    """A backend that reports neither a metric nor a cause is itself the finding."""
    res = verify_winner(_Backend(_M(0.0, "")), _Spec(), _Winner(), [1], lambda _: None)
    assert "no failure_reason" in res["remeasure_failure"]


def test_a_successful_remeasure_carries_no_failure_text():
    res = verify_winner(_Backend(_M(6.4, "", [1, 2, 3])), _Spec(), _Winner(),
                        [1, 2, 3], lambda _: None)
    assert res["verdict"] == "verified", res
    assert res["remeasure_failure"] == ""


def test_a_reason_present_on_a_SUCCESSFUL_remeasure_is_not_reported():
    """Only a zero metric means the re-measure failed; a stray reason must not
    turn a good verification into a confusing one."""
    res = verify_winner(_Backend(_M(6.4, "harmless note", [1, 2, 3])), _Spec(),
                        _Winner(), [1, 2, 3], lambda _: None)
    assert res["remeasure_failure"] == ""


def test_a_raising_backend_still_explains_itself():
    class _Boom(_Backend):
        def measure(self, *a):
            raise RuntimeError("nrt init failed")

    logs: list[str] = []
    res = verify_winner(_Boom(None), _Spec(), _Winner(), [1], logs.append)
    assert res["verdict"] == "ungraded"
    assert "nrt init failed" in res["remeasure_failure"]
    assert any("nrt init failed" in m for m in logs)
