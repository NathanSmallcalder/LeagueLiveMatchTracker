import os
import csv
import json
import requests
import socket
import torch
import torch.nn as nn
from flask import Flask, render_template, jsonify
import urllib3
import joblib
import pandas as pd
import numpy as np
import random
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Disable SSL warnings only for localhost
_is_localhost = socket.gethostbyname(socket.gethostname()) in ('127.0.0.1', '::1')
if _is_localhost:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

# Rate limiting - 60 requests per minute per IP
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["60 per minute"],
    storage_uri="memory://"
)

# --- MODEL DEFINITION ---
class LoLLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        out = self.fc(last_hidden)
        return out

LoLNet = LoLLSTM

# --- GLOBAL APP STATE ---
FEATURE_COLUMNS = []
CONTINUOUS_COLS = []
ITEM_DATA = {}
MODELS = {}
SCALERS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XP_TABLE = {
    1: 0, 2: 280, 3: 660, 4: 1140, 5: 1720, 6: 2400, 7: 3180, 8: 4060,
    9: 5040, 10: 6120, 11: 7300, 12: 8580, 13: 9960, 14: 11440,
    15: 13020, 16: 14700, 17: 16480, 18: 18360
}

# Champion pools for simulator (top, jungle, mid, bot, support per team)
SIM_CHAMPION_POOLS = {
    "TOP": ["Darius", "Garen", "Malphite", "Camille", "Mordekaiser", "Ornn", "Jax", "Aatrox"],
    "JUNGLE": ["LeeSin", "Vi", "Hecarim", "Elise", "Ekko", "Kayn", "RekSai", "JarvanIV"],
    "MIDDLE": ["Ahri", "Zed", "Yasuo", "Syndra", "Orianna", "Viktor", "Leblanc", "Cassiopeia"],
    "BOTTOM": ["Jinx", "MissFortune", "Aphelios", "Tristana", "Vayne", "KaiSa", "Ezreal", "Samira"],
    "UTILITY": ["Thresh", "Leona", "Nautilus", "Morgana", "Lux", "Janna", "Lulu", "Rakan"]
}

SIM_ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

CS_PER_MINUTE_BY_ROLE = {
    "TOP": (8, 12),
    "JUNGLE": (5, 8),
    "MIDDLE": (9, 13),
    "BOTTOM": (10, 14),
    "UTILITY": (1, 3)
}

ITEM_PRICES = {}
SIM_COMMON_ITEMS = [
    {"itemID": 1036, "name": "Long Sword", "cost": 350},
    {"itemID": 1037, "name": "Amplifying Tome", "cost": 435},
    {"itemID": 1038, "name": "Needlessly Large Rod", "cost": 1250},
    {"itemID": 1039, "name": "Hunter's Talisman", "cost": 350},
    {"itemID": 1040, "name": "Dagger", "cost": 300},
    {"itemID": 1041, "name": "Ruby Crystal", "cost": 400},
    {"itemID": 1052, "name": "Amplifying Tome", "cost": 435},
    {"itemID": 1053, "name": "Null-Magic Mantle", "cost": 450},
    {"itemID": 1054, "name": "Refillable Potion", "cost": 150},
    {"itemID": 1055, "name": "Corrupting Potion", "cost": 250},
    {"itemID": 2052, "name": "Poro-Snax", "cost": 0},
    {"itemID": 3006, "name": "Berserker's Greaves", "cost": 300},
    {"itemID": 3011, "name": "Sorcerer's Shoes", "cost": 350},
    {"itemID": 3111, "name": "Mercury's Treads", "cost": 350},
    {"itemID": 3047, "name": "Plated Steelcaps", "cost": 350},
    {"itemID": 3070, "name": "Tear of the Goddess", "cost": 400},
    {"itemID": 3074, "name": "Ravenous Hydra", "cost": 330},
    {"itemID": 3089, "name": "Rabadon's Deathcap", "cost": 1100},
    {"itemID": 3115, "name": "Nashor's Tooth", "cost": 500},
    {"itemID": 3116, "name": "Rylai's Crystal Scepter", "cost": 450},
    {"itemID": 3151, "name": "Ionian Boots of Lucidity", "cost": 350},
    {"itemID": 3158, "name": "Ionian Boots of Lucidity", "cost": 350},
    {"itemID": 3504, "name": "Ardent Censer", "cost": 250},
    {"itemID": 3864, "name": "Mobility Boots", "cost": 350},
    {"itemID": 6676, "name": "The Collector", "cost": 525},
    {"itemID": 6653, "name": "Luden", "cost": 340},
    {"itemID": 6672, "name": "Infinity Edge", "cost": 725},
    {"itemID": 6673, "name": "Kraken Slayer", "cost": 300},
    {"itemID": 6675, "name": "Navori", "cost": 300},
    {"itemID": 6691, "name": "BotRK", "cost": 300},
    {"itemID": 6692, "name": "Cannon", "cost": 300},
    {"itemID": 6693, "name": "PD", "cost": 300},
    {"itemID": 6694, "name": "RFC", "cost": 300},
    {"itemID": 6695, "name": "KS", "cost": 300},
    {"itemID": 6696, "name": "IE", "cost": 725},
    {"itemID": 6697, "name": "GM", "cost": 300},
    {"itemID": 6698, "name": "BT", "cost": 300},
    {"itemID": 6699, "name": "LDR", "cost": 300},
    {"itemID": 6700, "name": "MR", "cost": 300},
    {"itemID": 6701, "name": "QSS", "cost": 300},
    {"itemID": 6702, "name": "GA", "cost": 300},
    {"itemID": 6703, "name": "DD", "cost": 300},
    {"itemID": 6609, "name": "Shadowflame", "cost": 900},
    {"itemID": 6616, "name": "Horizon", "cost": 280},
    {"itemID": 6641, "name": "Crown", "cost": 280},
    {"itemID": 6655, "name": "Luden's", "cost": 450},
    {"itemID": 6656, "name": "Liandry", "cost": 400},
    {"itemID": 6657, "name": "Archangel", "cost": 300},
]

ITEM_BY_ROLE = {
    "TOP": [3074, 3047, 3864, 3006, 3111],
    "JUNGLE": [1039, 3047, 3864, 3006, 3111],
    "MIDDLE": [6655, 6609, 6616, 6641, 3089],
    "BOTTOM": [6672, 6673, 6676, 6675, 6692],
    "UTILITY": [3504, 3158, 3011, 3070, 3116],
}

def get_sim_item_price(item_id):
    if not ITEM_PRICES:
        for item in SIM_COMMON_ITEMS:
            ITEM_PRICES[item["itemID"]] = item["cost"]
    return ITEM_PRICES.get(item_id, 0)

# Internal Simulator State
simulator_active = False
simulator_tick = 0
sim_dragon_count = 0
sim_baron_count = 0
SIM_START_OFFSET = 300
SIM_MAX_TICKS = 210

def init_simulator_state():
    global sim_players, sim_events, sim_dragon_count, simulator_tick, sim_baron_count, sim_event_id
    
    blue_champs = {}
    red_champs = {}
    for i, role in enumerate(SIM_ROLE_ORDER):
        blue_champs[role] = random.choice(SIM_CHAMPION_POOLS[role])
        red_champs[role] = random.choice(SIM_CHAMPION_POOLS[role])
    
    starting_gold = {
        "TOP": 500, "JUNGLE": 450, "MIDDLE": 500, "BOTTOM": 500, "UTILITY": 400
    }
    
    starting_cs = {
        "TOP": 35, "JUNGLE": 0, "MIDDLE": 42, "BOTTOM": 38, "UTILITY": 0
    }
    
    starting_items = {
        "TOP": [1036],
        "JUNGLE": [1039],
        "MIDDLE": [1052],
        "BOTTOM": [1040],
        "UTILITY": [2031]
    }
    
    starting_level = {
        "TOP": 4, "JUNGLE": 4, "MIDDLE": 4, "BOTTOM": 4, "UTILITY": 3
    }
    
    sim_players = []
    for i, role in enumerate(SIM_ROLE_ORDER):
        sim_players.append({
            "summonerName": f"B_{role}",
            "championName": blue_champs[role],
            "team": "ORDER",
            "position": role,
            "level": starting_level[role],
            "items": [{"itemID": item} for item in starting_items[role]],
            "scores": {"kills": 0, "deaths": 0, "assists": 0, "creepScore": starting_cs[role]},
            "gold": starting_gold[role],
            "total_gold_spent": 0
        })
    
    for i, role in enumerate(SIM_ROLE_ORDER):
        sim_players.append({
            "summonerName": f"R_{role}",
            "championName": red_champs[role],
            "team": "CHAOS",
            "position": role,
            "level": starting_level[role],
            "items": [{"itemID": item} for item in starting_items[role]],
            "scores": {"kills": 0, "deaths": 0, "assists": 0, "creepScore": starting_cs[role]},
            "gold": starting_gold[role],
            "total_gold_spent": 0
        })
    
    sim_events = []
    sim_event_id = 1
    sim_dragon_count = 0
    sim_baron_count = 0
    simulator_tick = 0

# --- STARTUP ROUTINES ---
def fetch_datadragon_items():
    print("Fetching latest Data Dragon items...")
    try:
        versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
        latest = versions[0]
        items_resp = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/item.json").json()
        
        global ITEM_DATA
        ITEM_DATA = {}
        for item_id, item_info in items_resp['data'].items():
            ITEM_DATA[int(item_id)] = item_info.get('gold', {}).get('total', 0)
        print(f"Loaded {len(ITEM_DATA)} items from Data Dragon patch {latest}.")
    except Exception as e:
        print(f"Failed to fetch Data Dragon items: {e}")

def load_feature_template():
    print("Loading feature vector template from df_wide.csv...")
    global FEATURE_COLUMNS, CONTINUOUS_COLS
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'df_wide.csv')
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            FEATURE_COLUMNS = next(reader)
            
        keywords = ['gold', 'xp', 'cs', 'level', 'kills', 'deaths', 'diff', 'team_']
        CONTINUOUS_COLS = [c for c in FEATURE_COLUMNS if any(x in c.lower() for x in keywords)]
        
        print(f"Loaded {len(FEATURE_COLUMNS)} feature columns.")
    except Exception as e:
        print(f"Failed to load feature template: {e}")

def get_model_and_scaler(target_min):
    if target_min not in MODELS:
        print(f"Loading model and scaler for {target_min}m...")
        base_dir = os.path.dirname(__file__)
        scaler_path = os.path.join(base_dir, f'scaler_{target_min}m.pkl')
        model_path = os.path.join(base_dir, f'lol_model_{target_min}m.pth')
        
        if os.path.exists(scaler_path) and os.path.exists(model_path):
            scaler = joblib.load(scaler_path)
            # Input dim = total columns - 3 (match_id, minute, winning_team)
            drop_cols = ['winning_team', 'match_id', 'minute']
            feature_count = len([c for c in FEATURE_COLUMNS if c not in drop_cols])
            
            model = LoLNet(feature_count)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
            model.eval()
            
            SCALERS[target_min] = scaler
            MODELS[target_min] = model
        else:
            print(f"Files not found for {target_min}m")
            return None, None
            
    return MODELS[target_min], SCALERS[target_min]

with app.app_context():
    fetch_datadragon_items()
    load_feature_template()

# --- INTERNAL SIMULATOR ---

@app.route('/api/start_simulator', methods=['POST'])
@limiter.limit("10 per minute")
def start_simulator():
    global simulator_active
    try:
        init_simulator_state()
        simulator_active = True
        return jsonify({"status": "started", "message": "Match simulator started successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stop_simulator', methods=['POST'])
@limiter.limit("10 per minute")
def stop_simulator():
    global simulator_active
    simulator_active = False
    # Reset simulator state to initial values
    init_simulator_state()
    return jsonify({"status": "stopped", "message": "Simulator stopped and state reset"})

def generate_simulated_data():
    global simulator_tick, sim_players, sim_events, sim_event_id, sim_dragon_count, sim_baron_count, simulator_active
    
    if not simulator_active:
        return None
    
    simulator_tick += 1
    
    if simulator_tick > SIM_MAX_TICKS:
        init_simulator_state()
        simulator_tick = 1
    
    game_time = SIM_START_OFFSET + (simulator_tick * 10)
    game_minutes = game_time / 60.0
    
    for p in sim_players:
        role = p.get('position', '')
        gold = p.get('gold', 0)
        
        cs_min, cs_max = CS_PER_MINUTE_BY_ROLE.get(role, (8, 12))
        cs_rate = random.uniform(cs_min, cs_max) / 6.0
        cs_gained = max(0, int(random.gauss(cs_rate, cs_rate * 0.3)))
        p['scores']['creepScore'] += cs_gained
        gold += cs_gained * 21
        
        level_progression = min(18, max(1, int(game_minutes / 2.5) + 3))
        p['level'] = min(18, max(p['level'], level_progression))
        
        passive_gold = 2.0 + (game_minutes * 0.3)
        gold += passive_gold
        
        death_timer = victim.get('respawnTimer', 0) if 'victim' in locals() else 0
        if p['scores']['deaths'] > len(getattr(p, 'death_times', [])):
            gold_loss = 50 + int(gold * 0.05)
            gold = max(300, gold - gold_loss)
        
        if random.random() < 0.08 and len(p['items']) < 6:
            item_pool = ITEM_BY_ROLE.get(role, ITEM_BY_ROLE["MIDDLE"])
            available_items = [i for i in item_pool if i not in [x['itemID'] for x in p['items']]]
            if available_items and gold > 500:
                new_item = random.choice(available_items)
                item_cost = get_sim_item_price(new_item)
                if gold >= item_cost:
                    p['items'].append({"itemID": new_item})
                    p['total_gold_spent'] += item_cost
                    gold -= item_cost
        
        p['gold'] = gold
    
    blue_kills = sum(p['scores']['kills'] for p in sim_players if p['team'] == 'ORDER')
    red_kills = sum(p['scores']['kills'] for p in sim_players if p['team'] == 'CHAOS')
    
    blue_bounty = 300 if blue_kills > red_kills else 0
    red_bounty = 300 if red_kills > blue_kills else 0
    
    blue_deaths = sum(p['scores']['deaths'] for p in sim_players if p['team'] == 'ORDER')
    red_deaths = sum(p['scores']['deaths'] for p in sim_players if p['team'] == 'CHAOS')
    
    for p in sim_players:
        kills = p['scores']['kills']
        assists = p['scores']['assists']
        deaths = p['scores']['deaths']
        cs = p['scores']['creepScore']
        items_value = sum(get_sim_item_price(i['itemID']) for i in p['items'])
        
        kill_assist_gold = (kills * 300) + (assists * 150)
        
        p['scores']['total_gold'] = 500 + (cs * 21) + kill_assist_gold + items_value
    
    if random.random() < 0.15:
        killer = random.choice(sim_players)
        victim = random.choice([x for x in sim_players if x['team'] != killer['team']])
        
        killer['scores']['kills'] += 1
        killer['gold'] += 300
        
        victim_kills = victim['scores']['kills']
        victim_deaths = victim['scores']['deaths']
        victim_bounty = 300
        if victim_kills > victim_deaths + 2:
            victim_bounty = 600
        elif victim_kills > victim_deaths:
            victim_bounty = 450
        
        victim['scores']['deaths'] += 1
        victim['gold'] = max(300, victim['gold'] - victim_bounty)
        
        assister_chance = random.random()
        if assister_chance < 0.7:
            for ally in sim_players:
                if ally['team'] == killer['team'] and ally != killer:
                    ally['scores']['assists'] += 1
                    ally['gold'] += 150
                    if assister_chance < 0.3:
                        break
        
        sim_events.append({
            "EventID": sim_event_id,
            "EventName": "ChampionKill",
            "EventTime": game_time,
            "KillerName": killer['summonerName'],
            "VictimName": victim['summonerName'],
            "Assisters": []
        })
        sim_event_id += 1
    
    if simulator_tick > 60 and random.random() < 0.08 and sim_dragon_count < 8:
        objective_team = random.choice(["ORDER", "CHAOS"])
        objective_player = random.choice([p for p in sim_players if p['team'] == objective_team])['summonerName']
        sim_events.append({
            "EventID": sim_event_id,
            "EventName": "DragonKill",
            "EventTime": game_time,
            "KillerName": objective_player
        })
        sim_dragon_count += 1
        sim_event_id += 1
    
    if game_time > 900 and sim_baron_count < 3 and random.random() < 0.05:
        objective_team = random.choice(["ORDER", "CHAOS"])
        objective_player = random.choice([p for p in sim_players if p['team'] == objective_team])['summonerName']
        sim_events.append({
            "EventID": sim_event_id,
            "EventName": "BaronKill",
            "EventTime": game_time,
            "KillerName": objective_player
        })
        sim_baron_count += 1
        sim_event_id += 1
    
    if random.random() < 0.05:
        objective_team = random.choice(["ORDER", "CHAOS"])
        objective_player = random.choice([p for p in sim_players if p['team'] == objective_team])['summonerName']
        sim_events.append({
            "EventID": sim_event_id,
            "EventName": "TurretKilled",
            "EventTime": game_time,
            "KillerName": objective_player
        })
        sim_event_id += 1
    
    return {
        "gameData": {"gameTime": game_time},
        "allPlayers": sim_players,
        "events": {"Events": sim_events}
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/live_prediction')
@limiter.limit("30 per minute")
def live_prediction():
    global simulator_active
    try:
        # If simulator active, generate data internally
        if simulator_active:
            data = generate_simulated_data()
            if data is None:
                return jsonify({"error": "Simulator not active"}), 400
        else:
            # 1. Fetch live client data
            try:
                # Try HTTPS first (Real League Client)
                riot_url = os.environ.get('RIOT_API_URL', 'https://127.0.0.1:2999/liveclientdata/allgamedata')
                try:
                    resp = requests.get(riot_url, verify=False, timeout=2)
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
                    # Fallback to HTTP (for our local Match Simulator)
                    fallback_url = riot_url.replace('https://', 'http://')
                    resp = requests.get(fallback_url, verify=False, timeout=2)
                
                if resp.status_code != 200:
                    return jsonify({"error": "Game not found or not started"}), 404
                data = resp.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                return jsonify({"error": "League of Legends client is not running or not in game"}), 404
        
        game_time = data.get('gameData', {}).get('gameTime', 0)
        time_mins = game_time / 60.0
        
        # Determine model to use
        if time_mins < 10: target_min = 5
        elif time_mins < 15: target_min = 10
        elif time_mins < 20: target_min = 15
        else: target_min = 20
        
        model, scaler = get_model_and_scaler(target_min)
        if not model:
            return jsonify({"error": f"Model for {target_min}m not found"}), 500
        
        # Initialize raw feature dict with zeros
        features = {col: 0.0 for col in FEATURE_COLUMNS}
        
        # Track team stats from events
        blue_stats = {"kills": 0, "towers": 0, "dragons": 0, "barons": 0, "inhibitors": 0, "heralds": 0, "void_grubs": 0}
        red_stats = {"kills": 0, "towers": 0, "dragons": 0, "barons": 0, "inhibitors": 0, "heralds": 0, "void_grubs": 0}
        
        player_team_map = {} # summonerName -> "blue" / "red"
        
        # Map players to slots (0-4 Blue, 5-9 Red)
        blue_idx, red_idx = 0, 5
        
        blue_gold_total = 0
        red_gold_total = 0
        
        for p in data.get('allPlayers', []):
            team = "blue" if p.get('team') == 'ORDER' else "red"
            full_name = p['summonerName']
            game_name = full_name.split('#')[0] if '#' in full_name else full_name
            player_team_map[full_name] = team
            player_team_map[game_name] = team
            
            slot = blue_idx if team == "blue" else red_idx
            if team == "blue": blue_idx += 1
            else: red_idx += 1
            
            slot = min(9, max(0, slot))
            
            # Simplified Gold: Starting gold + CS value + kill rewards
            # Base ~500 + cs*20 + kills*300 + assists*150
            cs = p.get('scores', {}).get('creepScore', 0)
            kills = p.get('scores', {}).get('kills', 0)
            assists = p.get('scores', {}).get('assists', 0)
            
            sim_total_gold = p.get('scores', {}).get('total_gold', 0)
            if sim_total_gold > 0:
                total_gold = sim_total_gold
            else:
                items_value = sum(ITEM_DATA.get(i['itemID'], 0) for i in p.get('items', []))
                total_gold = 500 + (cs * 21) + (kills * 300) + (assists * 150) + items_value
            
            if team == "blue": blue_gold_total += total_gold
            else: red_gold_total += total_gold
            
            level = p.get('level', 1)
            xp = XP_TABLE.get(level, XP_TABLE.get(18))
            scores = p.get('scores', {})
            
            # Assign player stats to features
            if f'total_gold_{slot}' in features: features[f'total_gold_{slot}'] = total_gold
            if f'xp_{slot}' in features: features[f'xp_{slot}'] = xp
            if f'level_{slot}' in features: features[f'level_{slot}'] = level
            if f'kills_{slot}' in features: features[f'kills_{slot}'] = scores.get('kills', 0)
            if f'deaths_{slot}' in features: features[f'deaths_{slot}'] = scores.get('deaths', 0)
            if f'assists_{slot}' in features: features[f'assists_{slot}'] = scores.get('assists', 0)
            if f'cs_{slot}' in features: features[f'cs_{slot}'] = scores.get('creepScore', 0)
            if f'jungle_cs_{slot}' in features: features[f'jungle_cs_{slot}'] = 0
            
            champ_name = p.get('championName', '')
            champ_col = f'champion_{champ_name}_{slot}'
            if champ_col in features:
                features[champ_col] = 1.0
        
        # Parse Events
        for ev in data.get('events', {}).get('Events', []):
            ev_name = ev.get('EventName')
            killer = ev.get('KillerName')
            team = player_team_map.get(killer)
            if not team: continue
            
            stats = blue_stats if team == "blue" else red_stats
            
            if ev_name == 'ChampionKill': stats['kills'] += 1
            elif ev_name == 'TurretKilled': stats['towers'] += 1
            elif ev_name == 'DragonKill': stats['dragons'] += 1
            elif ev_name == 'BaronKill': stats['barons'] += 1
            elif ev_name == 'InhibitorKilled': stats['inhibitors'] += 1
            elif ev_name == 'HeraldKill': stats['heralds'] += 1
            elif ev_name == 'HordeKill': stats['void_grubs'] += 1
        
        # Match level features (Differential: Blue - Red)
        if 'team_kills' in features: features['team_kills'] = blue_stats['kills'] - red_stats['kills']
        if 'team_inhibitors' in features: features['team_inhibitors'] = blue_stats['inhibitors'] - red_stats['inhibitors']
        if 'team_towers' in features: features['team_towers'] = blue_stats['towers'] - red_stats['towers']
        if 'team_dragons' in features: features['team_dragons'] = blue_stats['dragons'] - red_stats['dragons']
        if 'team_barons' in features: features['team_barons'] = blue_stats['barons'] - red_stats['barons']
        if 'team_void_grubs' in features: features['team_void_grubs'] = blue_stats['void_grubs'] - red_stats['void_grubs']
        if 'team_heralds' in features: features['team_heralds'] = blue_stats['heralds'] - red_stats['heralds']
        
        # Macro Feature Engineering
        for i in range(5):
            gold_diff_col = f'gold_diff_lane_{i}'
            xp_diff_col = f'xp_diff_lane_{i}'
            if gold_diff_col in features:
                features[gold_diff_col] = features.get(f'total_gold_{i}', 0) - features.get(f'total_gold_{i+5}', 0)
            if xp_diff_col in features:
                features[xp_diff_col] = features.get(f'xp_{i}', 0) - features.get(f'xp_{i+5}', 0)
        
        # Soul point logic
        if 'at_soul_point' in features:
            features['at_soul_point'] = 1.0 if abs(features.get('team_dragons', 0)) >= 3 else 0.0
        
        # Create DataFrame for scaling
        df_live = pd.DataFrame([features])
        
        # Scale continuous features
        df_live[CONTINUOUS_COLS] = scaler.transform(df_live[CONTINUOUS_COLS])
        
        # Drop metadata columns
        drop_cols = ['winning_team', 'match_id', 'minute']
        feature_cols = [c for c in FEATURE_COLUMNS if c not in drop_cols]
        
        feature_values = df_live[feature_cols].values.astype('float32')
        
        seq_len = 5
        seq_input = np.tile(feature_values, (seq_len, 1))
        seq_input = seq_input.reshape(1, seq_len, -1).astype(np.float32)
        
        X_tensor = torch.tensor(seq_input).to(DEVICE)
        
        with torch.no_grad():
            red_win_prob = model(X_tensor).item()
        
        blue_win_prob = 1.0 - red_win_prob
        
        print(f"\n=== LIVE PREDICTION DEBUG ===")
        print(f"Game time: {game_time:.1f}s ({time_mins:.1f}m)")
        print(f"Model: {target_min}m")
        print(f"Blue kills={blue_stats['kills']}, towers={blue_stats['towers']}, dragons={blue_stats['dragons']}")
        print(f"Red kills={red_stats['kills']}, towers={red_stats['towers']}, dragons={red_stats['dragons']}")
        print(f"Blue gold total: {blue_gold_total}, Red gold total: {red_gold_total}")
        print(f"Player count: {len(data.get('allPlayers', []))}")
        print(f"Events count: {len(data.get('events', {}).get('Events', []))}")
        print(f"player_team_map: {player_team_map}")
        print(f"Blue win prob: {blue_win_prob:.4f}, Red win prob: {red_win_prob:.4f}")
        print(f"==============================\n")
        
        return jsonify({
            "time": game_time,
            "blue_prob": blue_win_prob,
            "red_prob": red_win_prob,
            "blue_gold": blue_gold_total,
            "red_gold": red_gold_total,
            "blue_stats": blue_stats,
            "red_stats": red_stats
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
