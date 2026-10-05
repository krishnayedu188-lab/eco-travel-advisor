# 🌍 Eco-Travel Advisor

A Rasa chatbot that helps travellers plan lower-carbon trips. It compares the emissions of train, coach, car and flight, recommends transport and nearby hotels based on the user's sustainability preference, and hands complex requests to a human advisor with the full conversation context.

MSc Artificial Intelligence – Advanced Conversational UI Design and Chatbot Development (BSBI / UCA).

## Features
- Adaptive multi-turn trip intake (Rasa form) with validation, typo correction and local city names (e.g. "Wien")
- Carbon estimates from the Climatiq API (free Estimate endpoint) with an offline fallback (UK Government factors)
- Weighted ranking of transport by carbon and price, adjusted to the user's sustainability level
- Real hotels near the main station from OpenStreetMap; eco-certifications shown only when verified on official registries
- Recognised carbon-offset programmes (reduce first, never "carbon neutral")
- Two-stage fallback and human handover with a full-context ticket (deleted after 30 days)
- Accessible web chat: quick replies, colour-coded cards with text labels, hotel carousel, handover banner, optional voice input

## Project structure
```
actions/actions.py                    Custom actions (APIs, ranking, validation, handover)
actions/data/                         Verified certifications, offset programmes, OpenStreetMap cache
data/nlu.yml, rules.yml, stories.yml  Training data and conversation rules
domain.yml, config.yml                Intents, slots, form, responses; NLU pipeline and policies
frontend/index.html                   Web chat interface
tests/                                Unit tests (pytest) and test stories (rasa test core)
results/                              Test reports and confusion matrices
```

## Setup
Requires **Python 3.10** (Rasa 3.6 does not support Python 3.11+).

```bash
conda create -n rasa python=3.10
conda activate rasa
pip install -r requirements.txt
cp .env.example .env      # add your Climatiq key (optional: the bot falls back to offline estimates)
rasa train
```

## Run (three terminals)
```bash
rasa run actions                                   # 1: action server (port 5055)
rasa run --cors "*"                                # 2: Rasa server (port 5005)
python -m http.server 3000 --directory frontend    # 3: web chat
```
Then open http://localhost:3000

## Tests
```bash
python -m pytest -v
rasa test nlu --nlu data/nlu.yml --cross-validation --folds 3 --out results/nlu_cv_after
rasa test core --stories tests/test_stories.yml --out results/core
```
Results: 26/26 unit tests passed · 11/11 test stories (50/50 actions) · NLU intent precision 0.68 (3-fold cross-validation)

## Data sources
- **Climatiq API**: emission factors (ADEME, BEIS, CO2 Emissiefactoren, EPA)
- **© OpenStreetMap contributors** (ODbL) via the Overpass API: hotel names and locations
- **Official certification registries** (Green Key, EU Ecolabel, etc.): checked manually, see `actions/data/verified_certifications.json`
- The Amadeus Self-Service API was decommissioned on 17 July 2026, so hotel data comes from OpenStreetMap instead.

## Privacy
- No names or contact details are requested; session IDs are random.
- The API key is stored in `.env`, which is never committed.
- Handover tickets are stored in `handover_queue/` (git-ignored) and deleted automatically after 30 days.

## Limitations
- Small, developer-written NLU dataset (around 110 examples)
- No live hotel prices; transport fares are rough estimates
- Human handover is simulated (tickets are saved, but there is no live advisor dashboard)