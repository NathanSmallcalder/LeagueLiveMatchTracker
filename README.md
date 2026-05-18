# League of Legends Live Win Probability Tracker

A real-time win probability predictor for League of Legends matches, built with PyTorch and Flask.

## Data

https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-match-interval-snapshots-2026

## Overview

This application connects to Riot's live client API to fetch real-time game data and uses a neural network to predict the probability of each team winning. It features a Hextech-themed web interface with live Chart.js updates showing win probability trends, gold lead, and objective tracking.

## Features

- **Real-time win probability prediction** using a PyTorch neural network
- **Live game tracking** via Riot's local client API
- **Interactive demo mode** with an internal game simulator (no active game required)
- **1,821 ML features** including per-player stats, champion matchups, lane differentials, and objectives
- **Hextech-themed UI** with live charts showing win probability, gold lead, kills, towers, dragons, and barons
- **Docker-ready** for easy deployment

## Quick Start

### Prerequisites

- Python 3.12+
- League of Legends client (optional - for live tracking)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/lol-tracker.git
cd lol-tracker
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser.

### Demo Mode

Click the **Run Demo** button to see the tracker in action with simulated game data. This runs entirely offline and does not require an active League game.

### Docker

```bash
docker build -t league-tracker .
docker run -p 5000:5000 --network=host league-tracker
```

## Tech Stack

- **Backend:** Python, Flask, PyTorch, Pandas, Requests
- **Frontend:** HTML/CSS/JS, Chart.js
- **Data:** Riot API (live client), Data Dragon (item data)
- **Deployment:** Docker

## The Model

The predictor is a **3-block residual neural network** trained on match intervals:

- **Architecture:** Input projection (1818 -> 256) + 3 ResidualBlocks (256) + output head (256 -> 128 -> 1)
- **Activation:** GELU with LayerNorm
- **Regularization:** Dropout (0.5 input, 0.3 residual, 0.4 head) and weight decay (AdamW, lambda=0.01)
- **Optimization:** AdamW with ReduceLROnPlateau scheduler and early stopping
- **Separate models** trained for 5m, 10m, 15m, 20m, and 25m time intervals
- **Weight initialization:** Xavier uniform

## Feature Engineering

The model uses 1,821 features per game state:

| Category | Count | Examples |
|----------|-------|----------|
| Team-level | 8 | team_kills, team_towers, team_dragons, team_barons, at_soul_point |
| Per-player | 80 | total_gold, xp, level, kills, deaths, assists, cs per slot 0-9 |
| Lane differentials | 10 | gold_diff_lane_0-4, xp_diff_lane_0-4 |
| Champion one-hot | 1710 | champion_Ahri, champion_Zed, etc. |
| Jungle CS | 10 | jungle_cs_0 to jungle_cs_9 |
| Inhibitors | 3 | team_inhibitors, team_void_grubs, team_heralds |

## Project Structure

```
lol-tracker/
├── app.py              # Flask application with live prediction logic
├── model.py            # Training script (ImprovedLoLNet architecture)
├── df_wide.csv         # Feature template (1821 columns, headers only)
├── lol_model_5m.pth    # Trained model weights (5 minute intervals)
├── lol_model_10m.pth
├── lol_model_15m.pth
├── lol_model_20m.pth
├── lol_model_25m.pth
├── scaler_5m.pkl       # Feature scalers for each time interval
├── scaler_10m.pkl
├── scaler_15m.pkl
├── scaler_20m.pkl
├── scaler_25m.pkl
├── templates/
│   └── index.html      # Hextech-themed frontend
├── Dockerfile          # Docker configuration
├── Procfile            # Render deployment configuration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Demo 

<img width="917" height="860" alt="image" src="https://github.com/user-attachments/assets/8ae3e623-1098-4a34-80ef-148dc01b8d09" />

## License

MIT License
