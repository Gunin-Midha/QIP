# Setting up the QIP alpha research system on your laptop

This guide takes you from nothing to a first research run. It takes about 30 minutes,
most of it creating accounts. Everything here is free.

You will need:

- A WorldQuant BRAIN account (your own; the system logs in as you).
- A Google account (for a free Gemini key).
- A Groq account (for a free Qwen key; no card needed).
- Python 3.11 or newer, and git.

The commands are for macOS or Linux. On Windows, use WSL (Ubuntu) and follow the same
steps; plain PowerShell needs different syntax for setting keys (see the end).

---

## 1. Get the code

```bash
git clone https://github.com/Gunin-Midha/QIP.git
cd QIP/alpha_research
```

Every command below is run from this `alpha_research/` folder.

## 2. Install Python packages

Use a virtual environment so nothing clashes with other projects:

```bash
python3 -m venv .venv
source .venv/bin/activate          # run this again in every new terminal
pip install -e ".[test]"
```

This installs pandas, numpy, scipy, pyyaml and pytest. Nothing else is needed for Gemini,
Qwen or BRAIN.

## 3. Check the install

```bash
PYTHONPATH=src python3 -m pytest -q
```

You should see about 579 tests pass in a few seconds. The tests never use the internet or
any key. If they fail, stop here and ask in the group.

## 4. Get your free API keys

| Key | Where | Steps |
|---|---|---|
| Gemini | aistudio.google.com | Sign in, **Get API key**, **Create API key**. Then open aistudio.google.com/rate-limit and check that `gemini-3.6-flash` is listed for free. |
| Groq (for Qwen and gpt-oss) | console.groq.com | Sign up, **API Keys**, **Create API Key**. Check that `qwen/qwen3.8-27b` and `gpt-oss-120b` are in **Models**. |
| NVIDIA (for Llama and DeepSeek) | build.nvidia.com | Sign in (may ask for phone verification), open any model, **Get API Key**. The key starts with `nvapi-`. |
| GitHub Models (optional, hosted GPT) | github.com/settings/tokens | **Fine-grained token**, permission **Models: read**. About 50 calls a day. |

Treat a key like a password. Never paste it into chat, a file in the repo, or a commit.

## 5. Put your login and keys into the terminal

Keys live only in the terminal session, never in a file. Run each line, then paste the
value and press Enter. Nothing appears while you paste; that is on purpose.

```bash
export BRAIN_EMAIL="your-brain-login@example.com"
read -s BRAIN_PASSWORD && export BRAIN_PASSWORD
read -s GEMINI_API_KEY && export GEMINI_API_KEY
read -s GROQ_API_KEY && export GROQ_API_KEY
read -s NVIDIA_API_KEY && export NVIDIA_API_KEY
read -s GITHUB_MODELS_TOKEN && export GITHUB_MODELS_TOKEN   # only if you made one
```

Type the line exactly as shown: the word after `read -s` is the variable's name, not your
password. You must do this again in every new terminal window.

To check one is set without printing it: `[ -n "$GROQ_API_KEY" ] && echo set`

## 6. Connect to BRAIN

Run these one at a time. Each saves its output under `captures/` (ignored by git, so it
stays on your laptop).

BRAIN may ask for biometric verification on every new login (our consultant account
does). The command then prints a link: open it in a browser, verify, come back to the
terminal and press Enter. Each command below logs in again, so expect one check per command.

```bash
PYTHONPATH=src python3 -m alpha_research.cli brain auth          # login works?
PYTHONPATH=src python3 -m alpha_research.cli brain options       # what your account allows
PYTHONPATH=src python3 -m alpha_research.cli brain operators     # compares operators with ours
PYTHONPATH=src python3 -m alpha_research.cli brain datafields \
    --region USA --universe TOP3000 --delay 1 \
    --out captures/datafields_usa_top3000_d1.json               # needed by later steps
```

The datafield download can take 10-30 minutes: the consultant catalog has more than
10,000 fields, so it goes one dataset at a time and prints progress. For every region
and delay in one login (one biometric check), run `PYTHONPATH=src python3
scripts/brain_discovery.py` instead. It saves every dataset list but downloads fields
for only 20 datasets per region/delay (the most used per category plus a few little-explored
ones; `--max-datasets N` changes it, 0 = all, which takes more than a day). Each dataset is
saved as it arrives, so Ctrl+C loses nothing; rerun it to fill in anything that failed.

`brain operators` should report no differences. If it lists operators that are "in seed
list, not on BRAIN" or "on BRAIN, not in seed list", your account level differs from ours:
send the output to Gunin rather than editing `config/fastexpr.yaml` yourself.

## 7. Test each model with one call

First check that every model in `config/agents.yaml` answers (one tiny call each). If one
fails with "model not found", list what the host serves and fix the id in the config:

```bash
PYTHONPATH=src python3 -m alpha_research.cli models --check
PYTHONPATH=src python3 -m alpha_research.cli models --host nvidia --filter deepseek
```

Then try a real generation with two of them:

```bash
PYTHONPATH=src python3 -m alpha_research.cli generate --model gemini-3.6-flash \
    --hypothesis "Stocks that fell sharply last week rebound next week" --n 3 \
    --settings config/generation_settings.example.yaml \
    --datafields captures/datafields_usa_top3000_d1.json --out captures/gen_gemini.json

PYTHONPATH=src python3 -m alpha_research.cli generate --model groq:qwen/qwen3.8-27b \
    --hypothesis "Stocks that fell sharply last week rebound next week" --n 3 \
    --settings config/generation_settings.example.yaml \
    --datafields captures/datafields_usa_top3000_d1.json --out captures/gen_qwen.json
```

Each should print `3 candidates, 0 rejected` (or close to it) and three formulas.

## 8. Run one simulation on BRAIN

```bash
PYTHONPATH=src python3 -m alpha_research.cli run --client real \
    --candidates captures/gen_gemini.json --data-dir data
```

This simulates the three Gemini formulas on BRAIN (a few minutes each) and prints a
decision for each: `accept`, `repair`, `reject` or `pending`.

## 9. Your first research run

This runs every agent together: the Director shares out simulations, Gemini and Qwen write
formulas and critique each other's, BRAIN simulates, the rule-based judges decide, and the
Repair Agent fixes what it can.

```bash
PYTHONPATH=src python3 -m alpha_research.cli research \
    --plan config/research_plan.example.yaml \
    --datafields captures/datafields_usa_top3000_d1.json \
    --max-simulations 6 --max-llm-calls 20 --per-round 3 --rounds 2
```

Keep the budgets this small at first. The free tiers allow only a few calls per minute,
so the run may stop early with `HTTP 429: Rate limit reached`. Short limits are waited
out automatically (you will see "rate-limited; waiting 9s"); a daily quota still stops the
run. The report is still written, and the next run learns from every result so far and
skips formulas already tried. Repairs queued in a stopped run are not carried over.

Results are stored on your laptop in `data/` (a database plus a report JSON per run).
They are not shared with the team and must never be committed.

To change the research ideas, copy `config/research_plan.example.yaml`, edit the
hypotheses, and pass your copy to `--plan`.

`config/research_plan.v2.yaml` covers every region and delay. It needs the catalogs from
`scripts/brain_discovery.py` in `captures/consultant_catalogs/` (not in git), and then no
`--datafields`. Set `enabled: false` on regions you don't want in a run. Each run works on
3 regions (the least covered so far) and the next run on the next 3, so cap a run at 20
alphas and just run it again:

```bash
PYTHONPATH=src python3 -m alpha_research.cli research --plan config/research_plan.v2.yaml \
    --max-simulations 20 --max-llm-calls 60 --per-round 5 --rounds 6
```

The plan's `seeds` (earlier alphas to improve) are simulated first. One of them needs the
option6 dataset at delay 1; fetch it once (one BRAIN login):

```bash
PYTHONPATH=src python3 scripts/brain_discovery.py --regions USA --datasets option6
```

---

## When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `zsh: not an identifier` | You typed your password in place of the variable name | Type `read -s BRAIN_PASSWORD` exactly, then paste the password after pressing Enter |
| Exit code 7, BRAIN login error | Wrong BRAIN email or password, or the variables are not set in this terminal | Redo step 5 in this terminal |
| `GEMINI_API_KEY is not set` (or `GROQ_...`) | The key is not set in this terminal | Redo step 5 |
| `HTTP 429: Rate limit reached` (run stopped) | A free tier's daily limit, or a per-minute limit that kept recurring | Try again later or tomorrow, or use a smaller `--max-llm-calls` |
| `HTTP 403: error code: 1010` | Cloudflare blocked the request | Pull the latest code (`git pull`); this is fixed |
| `HTTP 404` or "model not found" | The model id changed at Google or Groq | Check the model lists (step 4) and tell Gunin |
| Exit code 2, holdout or config error | A config file was changed | `git status` and undo local changes to `config/` |
| `ModuleNotFoundError` | The virtual environment is not active | `source .venv/bin/activate` |

## Ground rules

- Never put a password or key in a file, a commit or a chat message.
- Never commit `captures/` or `data/` (git ignores them already).
- Thresholds, holdout dates and the operator list are shared team settings in `config/`.
  Propose changes to Gunin instead of editing them locally.
- More detail: `alpha_research/README.md` (how to run everything) and
  `alpha_research/REPORT.md` (how it was built).

## Windows PowerShell 7 (if you are not using WSL)

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[test]"
$env:PYTHONPATH = "src"
$env:BRAIN_EMAIL = "your-brain-login@example.com"
$env:BRAIN_PASSWORD = (Read-Host -AsSecureString | ConvertFrom-SecureString -AsPlainText)
$env:GEMINI_API_KEY = (Read-Host -AsSecureString | ConvertFrom-SecureString -AsPlainText)
$env:GROQ_API_KEY = (Read-Host -AsSecureString | ConvertFrom-SecureString -AsPlainText)
$env:NVIDIA_API_KEY = (Read-Host -AsSecureString | ConvertFrom-SecureString -AsPlainText)
python -m alpha_research.cli brain auth
```

Then run the other commands without the `PYTHONPATH=src` prefix, and with `python` in
place of `python3`. Line continuations use a backtick (`` ` ``) instead of `\`.
