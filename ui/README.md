# STELLAR React UI

This UI provides a configuration page to set up and launch STELLAR experiments.

## Features

- Test Domain Config section with editable features file content
- Execution Parameters section:
  - single turn / multi turn
  - testing time
  - population size
- Evaluation Parameters section:
  - fitness functions
  - judge model
  - judge temperature
  - oracle threshold
- Start Experiment button that sends configuration payload to backend runner bridge
- Runs page with per-run status and console output

## Run locally

1. Install Node.js 18+ (includes npm)
2. Start the STELLAR runner bridge in a separate terminal:

   python /home/lev/Documents/testing/ast-fortiss-tum/STELLAR/scripts/experiment_runner_server.py

3. Install dependencies:

   npm install

4. Start development server:

   npm run dev

5. Build production bundle:

   npm run build

By default, the UI posts to `http://localhost:8000/experiments/start`.

## Backend bridge endpoints

- POST `/experiments/start` to launch a run
- GET `/experiments/runs` to list runs
- GET `/experiments/runs/<run_id>` to retrieve one run including logs
