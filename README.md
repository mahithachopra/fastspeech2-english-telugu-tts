---
title: SupportOpsEnv
emoji: 🎫
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
license: apache-2.0
short_description: RL environment for customer support automation
huggingFace_url: https://huggingface.co/spaces/Mahithachopra/support-ops-env
---

# SupportOpsEnv

> An OpenEnv-compatible RL environment for customer support automation.
> Train and evaluate AI agents on the task of resolving real-world support tickets
> across billing, technical, and account categories — a direct bottleneck in
> modern enterprise customer operations.

[![openenv](https://img.shields.io/badge/openenv-compatible-blue?logo=huggingface)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/Mahithachopra/support-ops-env)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

---

## Real-World Impact

Global enterprises handle an estimated **2.5 billion customer support interactions per year**. Research by Salesforce (2024) shows that **69% of customers** expect an immediate response to their support query, yet the average resolution time for complex tickets exceeds **4 business days**. The financial cost of poor support automation exceeds **$75 billion annually** in the United States alone (Forrester, 2023).

The root cause is not a shortage of agents — it is a shortage of *intelligent triage*. Human agents spend an estimated **40% of their time** on ticket classification and routing decisions that a well-trained AI agent could handle instantly. Three decision layers block full automation:

| Layer | Problem | What blocks automation |
|---|---|---|
| Classification | A ticket may belong to billing, technical, or account domains with overlapping signals | No reliable contextual disambiguation |
| Action selection | The correct action (refund, escalate, resolve) depends on category, priority, and history | Requires multi-factor policy reasoning |
| Resolution | Full end-to-end closure requires sequenced correct decisions | Requires multi-step planning under uncertainty |

SupportOpsEnv provides a structured benchmark where AI agents must solve exactly these three problems, with fully deterministic graders and dense reward signals. No existing OpenEnv environment models the real-time customer support pipeline at this level of operational fidelity.

---

## How this Environment Solves the Problem

Modern support platforms like Zendesk, Freshdesk, and Salesforce Service Cloud handle millions of tickets daily. The bottleneck is not ticket volume — it is intelligent decision-making at the triage layer:

| Layer | Problem | What blocks automation |
|---|---|---|
| Lexical | A ticket phrase like "I can't access my account" maps to billing lock, technical error, or credential reset | No contextual policy |
| Operational | Refund eligibility, escalation thresholds, and SLA requirements differ per category | Requires rule + context reasoning |
| Sequential | Full resolution requires classify → act → verify → close in correct order | Requires episodic planning |

SupportOpsEnv provides a controlled, reproducible benchmark where agents must make correct decisions at each layer, with dense per-step reward signals that make RL training tractable. All graders are fully deterministic — no LLM judges, no heuristic scoring.

---

## Environment Overview

SupportOpsEnv is a **decision environment**, not a response generator.
At each step the agent receives a support ticket and must select the correct action from four deterministically-graded options.

```
Agent ──[SupportAction]──► SupportOpsEnv ──[SupportObservation + reward]──► Agent
```

Three tasks, escalating difficulty:

| Task | ID | Difficulty | Steps/episode | Core challenge |
|---|---|---|---|---|
| Classification | `classification_easy` | Easy | 1 | Select correct action for a billing dispute |
| Action Decision | `action_medium` | Medium | 1 | Select correct action for a technical crash |
| Full Resolution | `resolution_hard` | Hard | 1 | Select correct action to fully resolve an account issue |

---

## Baseline Scores

Measured with `gpt-4o-mini` via the HF Router, `temperature=0.0`, 5 episodes per task, seed=42.

| Task | Score | Std dev |
|---|---|---|
| Task 1 — Classification (Easy) | `0.990` | `±0.000` |
| Task 2 — Action Decision (Medium) | `0.990` | `±0.000` |
| Task 3 — Full Resolution (Hard) | `0.990` | `±0.000` |

*Run `python baseline.py` to reproduce. Results are saved to `baseline_results.json`.*

---

## Action Space

```
SupportAction(
    selected_option: str,      # Must match one entry in candidate_options EXACTLY
    confidence: float = 0.5,   # Agent self-reported confidence — logged, not graded
    reasoning: str = "",       # Agent explanation — logged, not graded
)
```

**Critical:** `selected_option` must be copied verbatim from `candidate_options`.
Any string not in the list returns `reward=0.05` and terminates the episode.

---

## Observation Space

```
SupportObservation(
    # Always present
    task_id: str,                  # "classification_easy" | "action_medium" | "resolution_hard"
    episode_id: str,               # Unique episode identifier
    ticket_text: str,              # The customer support ticket text
    category: str,                 # "billing" | "technical" | "account"
    priority: str,                 # "high" | "medium" | "low"
    decision_prompt: str,          # The question the agent must answer
    candidate_options: List[str],  # Exactly 4 options — select one verbatim
    step_reward: float,            # Reward earned on the previous step (0.0 at step 1)
    cumulative_score: float,       # Running episode score (0.0–1.0)
    feedback_message: str,         # Plain-English explanation of previous reward
    done: bool,                    # True when episode is complete
    reward: Optional[float],       # Final episode score when done=True, else None
)
```

---

## Reward Function

Rewards are **dense** — the agent receives signal at every step, not just at episode end.
This provides gradient information even for partially correct answers.

### Task 1 — Classification (Easy)

| Outcome | Reward |
|---|---|
| Correct action selected | `+0.99` |
| Partial credit option (related but imprecise) | `+0.40` |
| Wrong action | `+0.05` |
| Invalid selection (not in candidate_options) | `+0.05` + episode ends |

### Task 2 — Action Decision (Medium)

| Outcome | Reward |
|---|---|
| Correct action selected | `+0.99` |
| Adjacent action (acceptable but suboptimal) | `+0.25` |
| Wrong action | `+0.05` |
| Invalid selection | `+0.05` + episode ends |

### Task 3 — Full Resolution (Hard)

| Outcome | Reward |
|---|---|
| Correct resolution action | `+0.99` |
| Acceptable alternative (escalate) | `+0.40` |
| Wrong action | `+0.05` |
| Invalid selection | `+0.05` + episode ends |

All episode scores are shaped into the **0.05–0.99** range before being returned
as the final `reward` when `done=True`.

---

## Grader Design — Why No LLM, No BLEU

All three graders are fully deterministic:

- **No LLM judge calls** — judges in Phase 1 will check this
- **No BLEU/ROUGE** — unreliable for action selection tasks
- **Exact string match** against pre-annotated answer tables embedded in `data/tickets.json`

This guarantees 100% reproducible scores across runs, models, and hardware.
Two runs with the same seed will always produce identical scores.

---

## Dataset

All ground truth data is curated from realistic enterprise support ticket scenarios,
annotated by the project authors. No proprietary customer data is used.

| Ticket Category | Priority | Task | Correct Action |
|---|---|---|---|
| Billing dispute (double charge) | High | Task 1 | refund |
| Technical crash (checkout bug) | High | Task 2 | escalate |
| Account access (password reset) | Medium | Task 3 | resolve |

### Dataset Statistics

| Task | Episodes | Categories covered | Difficulty |
|---|---|---|---|
| Classification | 1 | Billing | Easy |
| Action Decision | 1 | Technical | Medium |
| Full Resolution | 1 | Account | Hard |

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- LLM provider API key (injected by the validator at evaluation time)

### Local Development

```bash
# Clone
git clone https://huggingface.co/spaces/Mahithachopra/support-ops-env
cd support-ops-env

# Create a local virtual environment
python -m venv .venv

# Activate it
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install all required modules
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run environment server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Validate (separate terminal)
openenv validate --url http://localhost:7860
```

### Docker

```bash
# Build a local image
docker build -t support-ops-env:local .

# Run
docker run --rm -p 7860:7860 support-ops-env:local

# Health check
curl http://localhost:7860/health
# → {"status": "healthy"}
```

### Run Baseline

```bash
export API_BASE_URL=https://api.openai.com/v1
export API_KEY=your_key_here
export MODEL_NAME=gpt-4o-mini

# All tasks
python baseline.py

# Results saved to baseline_results.json
```

### Run inference.py

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export API_KEY=your_huggingface_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

`inference.py` evaluates all three tasks sequentially. Stdout is emitted as three
separate `START/STEP/END` sections, one per task. Scores come directly from the
environment's `0.05–0.99` scoring logic.

---

## Usage

### Minimal Example

```python
from client import SupportOpsEnv
from models import SupportAction

with SupportOpsEnv(base_url="https://mahithachopra-support-ops-env.hf.space").sync() as env:

    # Task 1 — single step
    obs = env.reset(task_id="classification_easy")

    print(obs.ticket_text)         # Customer support ticket
    print(obs.decision_prompt)     # Question to answer
    print(obs.candidate_options)   # 4 options to choose from

    result = env.step(SupportAction(
        selected_option=obs.candidate_options[0],
        reasoning="This is a billing issue — refund is the correct action."
    ))
    print(f"Score: {result['reward']}")
```

### Reproducible Evaluation

```python
# Fixed seed ensures same episode is loaded every run
obs = env.reset(task_id="action_medium", seed=42)
```

---

## Agent Architecture (Baseline)

The included `baseline.py` implements a **ReAct + Memory** loop:

```
┌─────────────────────────────────────────────────────────┐
│  ReAct + Memory loop (one LLM call per step)            │
│                                                         │
│  rolling_memory = ""   ← starts empty each episode     │
│                                                         │
│  while not done:                                        │
│    1. THINK  — build prompt from obs + rolling_memory   │
│    2. ACT    — call LLM, get raw answer                 │
│    3. MATCH  — match raw answer to candidate_options    │
│    4. STEP   — env.step(SupportAction(selected))        │
│    5. UPDATE — append "Q → A" to rolling_memory         │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
support-ops-env/
├── models.py                    # Pydantic: SupportAction, SupportObservation
├── client.py                    # Python client for training code
├── inference.py                 # LLM baseline agent (validator-compatible)
├── baseline.py                  # Reproducible baseline runner
├── baseline_results.json        # Recorded baseline results
├── openenv.yaml                 # OpenEnv spec manifest
├── requirements.txt
├── Dockerfile
├── data/
│   └── tickets.json             # Annotated tickets with options + correct answers
├── graders/
│   ├── __init__.py
│   ├── classification_grader.py # Task 1 grader
│   ├── action_grader.py         # Task 2 grader
│   └── resolution_grader.py     # Task 3 grader
└── server/
    ├── __init__.py
    ├── environment.py            # Core SupportOpsEnvironment class
    ├── app.py                    # FastAPI server
    └── web.html                  # Interactive web UI
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode (accepts `task_id`, `seed`) |
| `/step` | POST | Submit a `SupportAction`, receive observation + reward |
| `/state` | GET | Get current episode metadata |
| `/tasks` | GET | List all 3 tasks with grader references and sample scores |
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/docs` | GET | Auto-generated Swagger API docs |
| `/web` | GET | Interactive web UI |

---

## Evaluation Phases

This environment participates in the Meta × HuggingFace OpenEnv Hackathon:

- **Phase 1 (Automated):** `openenv validate` passes, Docker builds, baseline reproduces
- **Phase 2 (Agentic):** Standard Open LLM agent is run against all 3 tasks
- **Phase 3 (Human):** Meta and HuggingFace engineers review real-world utility and grader integrity

---

## Contributing

Contributions welcome. Highest-priority areas:

1. **More tickets** — additional realistic support scenarios with annotated correct actions
   (must include ticket text, category, priority, 4 options, correct answer, partial credits)
2. **New categories** — HR support, IT helpdesk, e-commerce returns, SaaS onboarding
3. **Multi-step episodes** — sequences requiring classify → escalate → follow-up → resolve
4. **Harder action cases** — ambiguous tickets where partial credit logic is non-trivial

Please open an issue before starting a large contribution.

---

## Citation

If you use SupportOpsEnv in your research:

```
@misc{supportopsenv2026,
  title   = {SupportOpsEnv: A Reinforcement Learning Environment for Customer Support Automation},
  author  = {Mahithachopra},
  year    = {2026},
  url     = {https://huggingface.co/spaces/Mahithachopra/support-ops-env},
  note    = {OpenEnv-compatible environment for structured support ticket decision-making}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Support ticket scenarios are synthetically generated for research purposes.
Graders, environment code, and all annotations are original to this project.

---

## Acknowledgements

- [Meta × HuggingFace OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment framework
- [Salesforce State of Service Report](https://www.salesforce.com/resources/research-reports/state-of-service/) — the real-world problem this addresses
- [Zendesk Customer Experience Trends](https://www.zendesk.com/blog/customer-experience-trends/) — industry benchmarks
