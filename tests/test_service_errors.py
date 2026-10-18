"""When the AI service fails, that is not the agent failing.

A run can end because the AI service refused or failed every call -- the
host no longer supports a setting, the service is down, too many requests.
Such a run says nothing about the agent, and counting it as a fail would make
an outage look like a bad agent. These tests hold three rules:

- **Not a result.** A trial whose run ended in an AI-service error is not
  marked and not judged. Its folder is kept in ``_service_errors/`` and the
  trial is run again next time -- including an old one already marked as a
  fail -- and the results tables leave it out.
- **Stop when the service is down.** After three such trials in a row, the
  sweep stops rather than repeating the same error down the whole list.
- **Test the connection first.** Before any trial runs, one tiny call is made
  on each route the agent may use, built exactly as the agent's own calls
  are. A route that refuses it stops the sweep before anything else is spent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_eval_harness import (
    FakeJudge,
    FakeRunner,
    fake_harness,
    finished_run,
)
from test_metrics import make_sweep
from test_model_client import StubTransport, client_for

from web_action_agent.budget import PREFLIGHT_LOG, read_calls, sweep_spend
from web_action_agent.config import Config
from web_action_agent.eval_harness import (
    RAN,
    SERVICE_ERROR,
    SERVICE_ERROR_DIR,
    SERVICE_ERROR_OUTCOMES,
    SERVICE_ERROR_STREAK,
    SERVICE_FAILED,
    VERDICT_FILE,
    HarnessError,
    Trial,
    trial_state,
)
from web_action_agent.judge import FAIL, PASS, JudgeVerdict
from web_action_agent.metrics import check_categories, load_sweep, success_rate
from web_action_agent.model_client import (
    ENDPOINT_UNAVAILABLE,
    MODEL_ERROR,
    PREFLIGHT_SCHEMA,
    RATE_LIMITED,
    ModelClientError,
    load_profile,
)
from web_action_agent.trace import COST_LOG, OUTCOME_FILE

#: The error the planner's host actually returned, in the run that found this.
HOST_REFUSED = (
    "call to openrouter (nvidia/nemotron-3-nano-30b-a3b) failed: Error code: 404 - "
    "No endpoints found for nvidia/nemotron-3-nano-30b-a3b."
)


def end_in(folder: Path, category: str, reason: str = HOST_REFUSED) -> None:
    """Rewrite a finished run so it ended with the given category.

    Args:
        folder: A finished run folder.
        category: How the run ended, such as ``model_error``.
        reason: Why, as the run would have written it.
    """
    path = folder / OUTCOME_FILE
    outcome = json.loads(path.read_text())
    outcome["outcome"].update(ok=False, category=category, reason=reason, answer=None)
    path.write_text(json.dumps(outcome))


class ServiceDown(FakeRunner):
    """Writes runs that ended in an AI-service error, for the trials named.

    Attributes:
        ran: The trials it ran, in order.
    """

    def __init__(self, category: str = MODEL_ERROR, only: set[str] | None = None) -> None:
        """Set up the runner.

        Args:
            category: The service error the runs end in.
            only: Trial names that hit it; None for every trial.
        """
        super().__init__()
        self._category = category
        self._only = only

    def __call__(self, trial: Trial, config: Config, folder: Path, browser: Any) -> None:
        """Run one trial, ending it in the service error if it is one of ours."""
        super().__call__(trial, config, folder, browser)
        if self._only is None or trial.name in self._only:
            end_in(folder, self._category)


# --- Not a result ----------------------------------------------------------------


@pytest.mark.unit
def test_the_three_service_errors_are_the_model_layers_own() -> None:
    assert set(SERVICE_ERROR_OUTCOMES) == {MODEL_ERROR, RATE_LIMITED, ENDPOINT_UNAVAILABLE}


@pytest.mark.unit
@pytest.mark.parametrize("category", [MODEL_ERROR, RATE_LIMITED, ENDPOINT_UNAVAILABLE])
def test_a_service_error_is_set_aside_not_marked_or_judged(tmp_path: Path, category: str) -> None:
    judge = FakeJudge(JudgeVerdict(FAIL, "it failed", "planning"))
    sweep = tmp_path / "sweep"
    first = "fixture-home/trial-1"

    report = fake_harness(
        sweep, ServiceDown(category, only={first}), trials=1, workers=1, judge=judge
    ).run()

    result = next(r for r in report.results if r.trial.name == first)
    assert result.action == SERVICE_ERROR
    assert result.verdict is None, "no mark"
    assert "404" in result.error, "and the service's own words are kept"
    assert judge.asked == 0, "the judge was never asked"
    assert not (sweep / first).exists(), "the folder was moved out of the way..."
    kept = sweep / SERVICE_ERROR_DIR / "fixture-home" / "trial-1-1"
    assert (kept / OUTCOME_FILE).exists(), "...and kept as evidence"
    assert not (kept / VERDICT_FILE).exists()
    assert "1 stopped by an AI-service error (not counted; run again)" in report.summary()


@pytest.mark.unit
def test_an_agent_failure_is_still_marked_as_a_fail(tmp_path: Path) -> None:
    # Running out of steps is the agent's own doing, and stays a result.
    judge = FakeJudge(JudgeVerdict(FAIL, "it wandered", "navigation"))
    report = fake_harness(
        tmp_path / "sweep", FakeRunner(agent_ok=False), trials=1, workers=1, judge=judge
    ).run()

    assert {r.action for r in report.results} == {RAN}
    assert {r.verdict.status for r in report.results if r.verdict} == {FAIL}


@pytest.mark.unit
def test_a_restart_runs_the_trial_again(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    first = "fixture-home/trial-1"
    fake_harness(sweep, ServiceDown(only={first}), trials=1, workers=1).run()

    again = FakeRunner()
    report = fake_harness(sweep, again, trials=1, workers=1).run()

    assert again.ran == [first], "only the trial the service stopped"
    assert next(r for r in report.results if r.trial.name == first).verdict.status == PASS


@pytest.mark.unit
def test_an_old_run_already_marked_as_a_fail_is_run_again(tmp_path: Path) -> None:
    # As the run that found this left things: marked fail, kind "planning",
    # though the agent never got an answer from its model.
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1, workers=1).run()
    old = sweep / "fixture-home" / "trial-1"
    end_in(old, MODEL_ERROR)
    verdict = json.loads((old / VERDICT_FILE).read_text())
    verdict.update(status=FAIL, category="planning", agent_outcome=MODEL_ERROR)
    (old / VERDICT_FILE).write_text(json.dumps(verdict))
    assert trial_state(old) == SERVICE_FAILED

    again = FakeRunner()
    fake_harness(sweep, again, trials=1, workers=1).run()

    assert again.ran == ["fixture-home/trial-1"]
    assert (sweep / SERVICE_ERROR_DIR / "fixture-home" / "trial-1-1" / VERDICT_FILE).exists()


@pytest.mark.unit
def test_the_tables_leave_out_an_old_service_error_marked_as_a_fail(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    # charlie-1 was marked a fail; say its run had really ended in a 404.
    folder = sweep / "charlie" / "trial-1"
    outcome = json.loads((folder / OUTCOME_FILE).read_text())
    outcome["outcome"] = {"category": MODEL_ERROR}
    (folder / OUTCOME_FILE).write_text(json.dumps(outcome))

    rows = load_sweep(sweep).rows
    row = next(r for r in rows if (r.task_id, r.trial) == ("charlie", 1))

    assert not row.marked, "not a result"
    # Round 1 is now alpha, bravo and delta, all passes.
    assert success_rate(rows).passed == 8 and success_rate(rows).marked == 10
    check_categories([load_sweep(sweep)])  # and it is not a fail missing its kind


# --- Stop when the service is down ------------------------------------------------


@pytest.mark.unit
def test_the_sweep_stops_after_three_service_errors_in_a_row(tmp_path: Path) -> None:
    runner = ServiceDown()
    report = fake_harness(tmp_path / "sweep", runner, trials=2, workers=1).run()

    assert len(runner.ran) == SERVICE_ERROR_STREAK == 3
    assert "the AI service failed 3 trials in a row" in report.stopped_by
    assert "404" in report.stopped_by, "saying what the service said"
    assert report.not_started == 10 - 3


@pytest.mark.unit
def test_a_good_trial_between_errors_resets_the_count(tmp_path: Path) -> None:
    # In suite order the trials go error, good, error, error, good: two in a
    # row at most, so the sweep never stops.
    down = {f"{t}/trial-1" for t in ("fixture-home", "fixture-browse", "fixture-article")}
    runner = ServiceDown(only=down)

    report = fake_harness(tmp_path / "sweep", runner, trials=1, workers=1).run()

    assert runner.ran == [
        "fixture-home/trial-1",
        "fixture-search/trial-1",
        "fixture-browse/trial-1",
        "fixture-article/trial-1",
        "fixture-shelves/trial-1",
    ]
    assert report.stopped_by == ""


# --- Test the connection first ----------------------------------------------------


class Refusing(StubTransport):
    """A transport that refuses every call, as the planner's host did."""

    def _create(self, **request: Any) -> Any:
        """Record the request, then refuse it with a 404."""
        self.requests.append(request)
        error = RuntimeError("Error code: 404 - No endpoints found")
        error.status_code = 404  # type: ignore[attr-defined]
        raise error


@pytest.mark.unit
def test_preflight_makes_one_call_on_a_text_only_profile() -> None:
    transport = StubTransport('{"ok": true}', prompt_tokens=20, completion_tokens=3)

    records = client_for(load_profile("local"), transport=transport).preflight()

    assert len(records) == 1 and len(transport.requests) == 1
    assert records[0].total_tokens == 23


@pytest.mark.unit
def test_preflight_tests_the_screenshot_route_too_where_there_is_one() -> None:
    transport = StubTransport('{"ok": true}')
    profile = load_profile("openrouter")

    client_for(profile, transport=transport).preflight()

    text, vision = transport.requests
    assert text["model"] == profile.route_for(vision=False).model
    assert vision["model"] == profile.route_for(vision=True).model
    image = vision["messages"][0]["content"][1]["image_url"]["url"]
    assert image.startswith("data:image/png;base64,")


@pytest.mark.unit
def test_preflight_asks_exactly_as_the_agent_does() -> None:
    # The settings that broke the real run -- a strict schema, a seed, and the
    # host pin with no fallback -- must all be in the test call.
    transport = StubTransport('{"ok": true}')

    client_for(load_profile("openrouter"), transport=transport).preflight()

    text = transport.requests[0]
    assert text["response_format"]["json_schema"]["strict"] is True
    assert text["response_format"]["json_schema"]["schema"] == PREFLIGHT_SCHEMA
    assert "seed" in text
    assert text["extra_body"]["provider"]["allow_fallbacks"] is False


@pytest.mark.unit
def test_preflight_names_the_route_that_refused() -> None:
    client = client_for(load_profile("openrouter"), transport=Refusing())

    with pytest.raises(ModelClientError) as caught:
        client.preflight()

    message = str(caught.value)
    assert "the text route (nvidia/nemotron-3-nano-30b-a3b) did not answer a test call" in message
    assert "no trial was started" in message
    assert "404" in message


@pytest.mark.unit
def test_a_reply_that_is_not_json_still_proves_the_route_works() -> None:
    # The screenshot route has no schema support, so a chatty reply is normal.
    transport = StubTransport("Sure! Here you go: ok", prompt_tokens=5, completion_tokens=5)

    records = client_for(load_profile("local"), transport=transport).preflight()

    assert len(records) == 1, "answered, and paid for"


class Checked:
    """A stand-in client whose connection test passes or fails, and is counted."""

    def __init__(self, fails: bool = False) -> None:
        """Set up whether the test fails."""
        self.fails = fails
        self.tested = 0

    def preflight(self) -> tuple[Any, ...]:
        """Pass or fail the connection test."""
        self.tested += 1
        if self.fails:
            raise ModelClientError(f"the text route did not answer a test call: {HOST_REFUSED}")
        real = client_for(load_profile("local"), transport=StubTransport('{"ok": true}'))
        return real.preflight()


@pytest.mark.unit
def test_a_failed_connection_test_stops_the_sweep_before_any_trial(tmp_path: Path) -> None:
    runner = FakeRunner()
    harness = fake_harness(tmp_path / "sweep", runner, trials=1)
    harness._client_factory = lambda config: Checked(fails=True)

    with pytest.raises(HarnessError, match="did not answer a test call"):
        harness.run()

    assert runner.ran == []


@pytest.mark.unit
def test_the_connection_test_is_logged_and_counted(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    harness = fake_harness(sweep, FakeRunner(), trials=1)
    harness._client_factory = lambda config: Checked()

    harness.run()

    assert len(read_calls(sweep / PREFLIGHT_LOG)) == 1
    # Every trial's own calls, plus the one connection test.
    trial_calls = sum(len(read_calls(log)) for log in sweep.rglob(COST_LOG))
    assert sweep_spend(sweep).calls == trial_calls + 1


@pytest.mark.unit
def test_no_connection_test_when_nothing_is_left_to_run(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1).run()
    client = Checked()
    harness = fake_harness(sweep, FakeRunner(), trials=1)
    harness._client_factory = lambda config: client

    harness.run()

    assert client.tested == 0


@pytest.mark.unit
def test_finished_run_helper_is_the_agents_success(tmp_path: Path) -> None:
    # Guards the helpers above: a plain finished run is not a service error.
    finished_run(tmp_path / "run")
    assert trial_state(tmp_path / "run") != SERVICE_FAILED
