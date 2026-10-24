"""The AI judge: who may judge, what it is shown, and what its replies become.

Three groups of tests.

**Who may judge.** The judge must come from a different family of models than
the planner, and the program must refuse to start otherwise. The family rule
is tested on real model ids, including the awkward ones (a vendor building on
another vendor's model), and the refusal is tested end to end through the
command line, because "enforced at startup" is a claim about the program, not
about a function.

**What it is shown.** The request is read back from a stand-in model service:
the task, what success looks like, the answer, the steps, the screenshots as
images, and temperature zero. A judge shown the wrong things gives confident
verdicts about the wrong things.

**What its replies become.** A good reply becomes a pass or a fail with its
sentence and category. Every malformed reply becomes a *judging error* -- never
a pass, and never a fail either. These are the tests that matter most: a
judge whose broken reply quietly counted as a pass would inflate every number
the project publishes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.benchmark import BenchmarkTask
from web_action_agent.config import Config
from web_action_agent.judge import (
    FAIL,
    FAILURE_CATEGORIES,
    JUDGING_ERROR,
    PASS,
    Judge,
    JudgeConfigError,
    JudgeVerdict,
    ModelFamily,
    check_independence,
    independence_problem,
    model_family,
    parse_verdict,
    same_family,
    verdict_schema,
)
from web_action_agent.model_client import ModelClient, ModelClientError, load_profile
from web_action_agent.trace_reader import ReplayedRun, read_run

TRACE = Path(__file__).resolve().parent / "fixtures" / "traces" / "search-and-open"

#: A tiny valid PNG: the judge only needs bytes it can attach.
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x9a\xd0\xa5"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class JudgeEndpoint:
    """A stand-in model service that answers every call with one fixed reply.

    Attributes:
        requests: Every request received, so a test can read what was sent.
    """

    def __init__(self, reply: dict[str, Any] | str | Exception) -> None:
        """Set up the reply.

        Args:
            reply: A mapping is sent as JSON, a string as-is, and an exception
                is raised instead of answering, as a failing service would.
        """
        self._reply = reply
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> SimpleNamespace:
        """Answer one call."""
        self.requests.append(request)
        if isinstance(self._reply, Exception):
            raise self._reply
        content = self._reply if isinstance(self._reply, str) else json.dumps(self._reply)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=2400, completion_tokens=60),
        )

    @property
    def request(self) -> dict[str, Any]:
        """dict[str, Any]: The one request this endpoint received."""
        assert len(self.requests) == 1, f"expected one call, saw {len(self.requests)}"
        return self.requests[0]

    @property
    def text_sent(self) -> str:
        """str: All the text the judge was shown, system prompt included."""
        parts: list[str] = []
        for message in self.request["messages"]:
            content = message["content"]
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.extend(p["text"] for p in content if p.get("type") == "text")
        return "\n".join(parts)

    @property
    def images_sent(self) -> int:
        """int: How many images the judge was shown."""
        return sum(
            1
            for message in self.request["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
            if part.get("type") == "image_url"
        )


def a_judge(reply: dict[str, Any] | str | Exception) -> tuple[Judge, JudgeEndpoint]:
    """Build a judge on the real judge profile, answering from a stand-in.

    Args:
        reply: What the stand-in service replies.

    Returns:
        tuple[Judge, JudgeEndpoint]: The judge, and the endpoint to inspect.
    """
    endpoint = JudgeEndpoint(reply)
    client = ModelClient(load_profile("openrouter-judge"), transport=endpoint, env={})
    return Judge(client), endpoint


def a_task(**overrides: Any) -> BenchmarkTask:
    """Build the task the committed example run attempted.

    Args:
        **overrides: Fields to replace.

    Returns:
        BenchmarkTask: The task.
    """
    data: dict[str, Any] = {
        "id": "northgate-membership",
        "site": "northgate",
        "difficulty": "multi_page_navigation",
        "start_url": "http://northgate.example/search.html",
        "instruction": "Find the record about membership and open it.",
        "verdicts": [
            {
                "method": "judge",
                "expected_outcome": "The membership application form is open.",
            }
        ],
    }
    data.update(overrides)
    return BenchmarkTask.model_validate(data)


def the_run() -> ReplayedRun:
    """Read the committed example run: four steps, ending on a form."""
    return read_run(TRACE)


def a_run_with_screenshots(tmp_path: Path, count: int) -> ReplayedRun:
    """The example run, with a picture saved for each of its first steps.

    Args:
        tmp_path: Where to save the pictures.
        count: How many steps get one.

    Returns:
        ReplayedRun: The run.
    """
    run = the_run()
    steps = []
    for index, step in enumerate(run.steps):
        if index < count:
            picture = tmp_path / f"step-{index}.png"
            picture.write_bytes(PNG)
            step = replace(step, observation=replace(step.observation, screenshot_on_disk=picture))
        steps.append(step)
    return replace(run, steps=tuple(steps))


# --- Model families -------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "family"),
    [
        ("nvidia/nemotron-3-nano-30b-a3b", ModelFamily("nvidia", "nemotron")),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", ModelFamily("nvidia", "nemotron")),
        ("google/gemini-3.8-flash", ModelFamily("google", "gemini")),
        ("anthropic/claude-sonnet-5.5", ModelFamily("anthropic", "claude")),
        ("openai/gpt-5.6-luna", ModelFamily("openai", "gpt")),
        ("qwen/qwen3-vl-235b", ModelFamily("qwen", "qwen")),
        ("Meta-Llama/Llama-4-Maverick", ModelFamily("meta-llama", "llama")),
        ("gpt-4o", ModelFamily("", "gpt")),  # an id with no vendor
    ],
)
def test_model_family_is_read_from_the_id(model_id: str, family: ModelFamily) -> None:
    assert model_family(model_id) == family


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("nvidia/nemotron-3-nano", "nvidia/nemotron-ultra", True),  # same line
        ("nvidia/nemotron-3-nano", "nvidia/llama-3.1-nemotron", True),  # same vendor
        ("meta-llama/llama-4", "nvidia/llama-4-tuned", True),  # same line, two vendors
        ("nvidia/nemotron-3-nano", "google/gemini-3.8-flash", False),
        ("nvidia/nemotron-3-nano", "anthropic/claude-sonnet-5.5", False),
        ("gpt-4o", "claude-3", False),  # no vendors to compare: lines differ
    ],
)
def test_same_family_is_cautious(a: str, b: str, same: bool) -> None:
    assert same_family(model_family(a), model_family(b)) is same


@pytest.mark.unit
def test_the_shipped_judge_is_independent_of_every_shipped_planner() -> None:
    judge = load_profile("openrouter-judge")
    for planner in ("openrouter", "nvidia", "local"):
        check_independence(load_profile(planner), judge)  # must not raise


@pytest.mark.unit
def test_a_judge_in_the_planners_family_is_refused_with_a_clear_error() -> None:
    with pytest.raises(JudgeConfigError) as raised:
        check_independence(load_profile("openrouter"), load_profile("nvidia"))
    message = str(raised.value)
    assert "nvidia/nemotron-3-nano-30b-a3b" in message, "it names the clashing models"
    assert "same family" in message
    assert "different family" in message, "and says what to do instead"


@pytest.mark.unit
def test_independence_is_checked_against_the_planners_vision_model_too(tmp_path: Path) -> None:
    # A planner whose text model is fine but whose vision model clashes.
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        "planner:\n"
        "  base_url: https://x.test/v1\n  api_key_env: WAA_X_KEY\n"
        "  text: {model: acme/widget-1, provider_order: [], supports_json_schema: true}\n"
        "  vision: {model: google/gemini-vision, provider_order: [], supports_json_schema: true}\n"
        "judge:\n"
        "  base_url: https://y.test/v1\n  api_key_env: WAA_Y_KEY\n"
        "  text: {model: google/gemini-3.8-flash, provider_order: [], supports_json_schema: true}\n"
    )
    config = Config(profile="planner", judge_profile="judge")
    problem = independence_problem(config, profiles_path=profiles)
    assert problem is not None and "google/gemini-vision" in problem


@pytest.mark.unit
def test_no_judge_configured_is_not_a_problem_at_startup() -> None:
    assert independence_problem(Config(judge_profile=None)) is None


@pytest.mark.unit
def test_from_config_refuses_a_same_family_judge() -> None:
    with pytest.raises(JudgeConfigError, match="same family"):
        Judge.from_config(Config(judge_profile="local"), env={})


@pytest.mark.unit
def test_from_config_refuses_when_no_judge_is_configured() -> None:
    with pytest.raises(JudgeConfigError, match="no judge is configured"):
        Judge.from_config(Config(judge_profile=None), env={})


@pytest.mark.unit
def test_the_program_refuses_to_start_with_a_same_family_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Through the real command line: the refusal happens before any run.
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_JUDGE_PROFILE": "nvidia"})
    started: list[str] = []
    monkeypatch.setattr(cli, "_do_one_task", lambda config, task, **_: started.append(task) or 0)

    result = CliRunner().invoke(cli.app, ["run", "find something"], catch_exceptions=False)

    assert result.exit_code == cli.EXIT_USAGE
    assert "same family" in result.output
    assert started == [], "no task was started"


@pytest.mark.unit
def test_the_program_starts_with_an_independent_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_JUDGE_PROFILE": "openrouter-judge"})
    started: list[str] = []
    monkeypatch.setattr(cli, "_do_one_task", lambda config, task, **_: started.append(task) or 0)

    result = CliRunner().invoke(cli.app, ["run", "find something"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert started == ["find something"]


# --- What the judge is shown ------------------------------------------------------


@pytest.mark.mock_llm
def test_the_judge_is_shown_the_task_the_goal_the_answer_and_the_steps() -> None:
    judge, endpoint = a_judge({"passed": True, "justification": "x", "category": "none"})
    judge.judge(a_task(), the_run())
    sent = endpoint.text_sent
    assert "Find the record about membership and open it." in sent, "the task"
    assert "The membership application form is open." in sent, "what success looks like"
    assert "Membership application" in sent, "the agent's answer"
    assert "fill" in sent and "click" in sent, "the steps it took"
    assert "Judge the evidence, not the claim" in sent, "and the instructions"


@pytest.mark.mock_llm
def test_the_judge_sees_the_last_three_screenshots(tmp_path: Path) -> None:
    judge, endpoint = a_judge({"passed": True, "justification": "x", "category": "none"})
    judge.judge(a_task(), a_run_with_screenshots(tmp_path, count=4))
    assert endpoint.images_sent == 3


@pytest.mark.mock_llm
def test_the_judge_runs_at_temperature_zero_with_the_schema() -> None:
    judge, endpoint = a_judge({"passed": True, "justification": "x", "category": "none"})
    judge.judge(a_task(), the_run())
    request = endpoint.request
    assert request["temperature"] == 0.0
    assert request["model"] == "google/gemini-3.8-flash"
    assert request["response_format"]["json_schema"]["schema"] == verdict_schema()


@pytest.mark.mock_llm
def test_a_task_without_a_written_goal_is_judged_from_the_task_alone() -> None:
    judge, endpoint = a_judge({"passed": True, "justification": "x", "category": "none"})
    task = a_task(verdicts=[{"method": "url_contains", "value": "form-step-1"}])
    judge.judge(task, the_run())
    assert "judge from the task alone" in endpoint.text_sent


@pytest.mark.mock_llm
def test_the_judge_is_told_when_the_final_page_was_never_captured() -> None:
    # Cut the run after its third step: a click onto a page nobody looked at.
    run = the_run()
    cut = replace(run, steps=run.steps[:3], outcome=None)
    judge, endpoint = a_judge({"passed": False, "justification": "x", "category": "planning"})
    judge.judge(a_task(), cut)
    assert "never captured" in endpoint.text_sent
    assert "interrupted before it finished" in endpoint.text_sent


# --- What its replies become ------------------------------------------------------


@pytest.mark.mock_llm
def test_a_pass_is_parsed_with_its_justification() -> None:
    judge, _ = a_judge(
        {"passed": True, "justification": "The application form is open.", "category": "none"}
    )
    verdict = judge.judge(a_task(), the_run())
    assert verdict.outcome == PASS and verdict.passed
    assert verdict.justification == "The application form is open."
    assert verdict.category is None
    assert verdict.model == "google/gemini-3.8-flash"
    assert verdict.record is not None, "and what the call cost"


@pytest.mark.mock_llm
@pytest.mark.parametrize("category", FAILURE_CATEGORIES)
def test_a_fail_is_parsed_with_its_category(category: str) -> None:
    judge, _ = a_judge(
        {"passed": False, "justification": "It opened the wrong record.", "category": category}
    )
    verdict = judge.judge(a_task(), the_run())
    assert verdict.outcome == FAIL and not verdict.passed
    assert verdict.category == category
    assert verdict.justification == "It opened the wrong record."


@pytest.mark.mock_llm
def test_a_pass_with_a_stray_category_is_still_a_pass() -> None:
    judge, _ = a_judge({"passed": True, "justification": "Done.", "category": "grounding"})
    verdict = judge.judge(a_task(), the_run())
    assert verdict.outcome == PASS
    assert verdict.category is None


@pytest.mark.mock_llm
@pytest.mark.parametrize(
    ("reply", "because"),
    [
        ("I think it passed!", "not a JSON object"),
        ('{"passed": "true", "justification": "x", "category": "none"}', "true or false"),
        ('{"passed": "false", "justification": "x", "category": "planning"}', "true or false"),
        ('{"justification": "x", "category": "none"}', "true or false"),
        ('{"passed": true, "justification": "", "category": "none"}', "no justification"),
        ('{"passed": true, "category": "none"}', "no justification"),
        ('{"passed": false, "justification": "x", "category": "luck"}', "'category' must be"),
        ('{"passed": false, "justification": "x"}', "'category' must be"),
        ('{"passed": false, "justification": "x", "category": "none"}', "no failure category"),
    ],
)
def test_a_malformed_reply_is_a_judging_error_not_a_pass(reply: str, because: str) -> None:
    judge, _ = a_judge(reply)
    verdict = judge.judge(a_task(), the_run())
    assert verdict.outcome == JUDGING_ERROR
    assert not verdict.passed
    assert because in verdict.error
    assert verdict.raw_reply == reply, "the reply is kept for a person to read"
    assert verdict.record is not None, "a wasted call still cost money"


@pytest.mark.mock_llm
def test_an_unreachable_judge_is_a_judging_error_not_a_pass() -> None:
    judge, _ = a_judge(ModelClientError("connection refused"))
    verdict = judge.judge(a_task(), the_run())
    assert verdict.outcome == JUDGING_ERROR
    assert "could not be reached" in verdict.error


@pytest.mark.unit
def test_parse_verdict_never_turns_a_broken_reply_into_a_pass() -> None:
    # The same rule, on the parser alone, across everything in the reply
    # that could plausibly be mistaken for "yes".
    for passed in ("true", "yes", 1, "1", None, [True]):
        data = {"passed": passed, "justification": "x", "category": "none"}
        assert parse_verdict(data, raw_reply="").outcome == JUDGING_ERROR, passed


@pytest.mark.unit
def test_a_verdict_survives_being_written_and_read_back() -> None:
    judge, _ = a_judge({"passed": False, "justification": "Wrong row.", "category": "grounding"})
    verdict = judge.judge(a_task(), the_run())
    again = JudgeVerdict.from_dict(json.loads(json.dumps(verdict.to_dict())))
    assert again == verdict


# --- Against the real judge ---------------------------------------------------------


@pytest.mark.live
def test_the_real_judge_marks_the_example_run() -> None:
    """One real call to the shipped judge, to prove the whole path works.

    Excluded from ordinary test runs and from continuous integration: it needs
    the OpenRouter key and spends a fraction of a cent. Run it with
    ``pytest -m live tests/test_judge.py``.
    """
    from web_action_agent.config import resolve_env

    env = resolve_env()
    if not env.get("WAA_OPENROUTER_API_KEY"):
        pytest.skip("WAA_OPENROUTER_API_KEY is not set")
    judge = Judge.from_config(Config(judge_profile="openrouter-judge"), env=env)

    verdict = judge.judge(a_task(), the_run())

    assert verdict.outcome in (PASS, FAIL), verdict.error or verdict.raw_reply
    assert verdict.justification
