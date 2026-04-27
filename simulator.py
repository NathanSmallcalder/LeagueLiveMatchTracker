"""
Match Simulator - Provides fake live game data for demo purposes
Time: 5 minute offset, runs for 40 minutes, then resets
"""
import random
from flask import Flask, jsonify

app = Flask(__name__)

# Simulation state
tick = 0
START_OFFSET = 300  # 5 mins in seconds
MAX_TICKS = 210  # ~35 mins of ticks (each tick = 10 game seconds)

# Initial players at 5 minute mark
players = [
    {"summonerName": "B_Top", "team": "ORDER", "position": "TOP", "level": 4, "items": [{"itemID": 1036}], "scores": {"kills": 1, "deaths": 0, "assists": 2, "creepScore": 35}},
    {"summonerName": "B_Jgl", "team": "ORDER", "position": "JUNGLE", "level": 4, "items": [{"itemID": 1037}], "scores": {"kills": 0, "deaths": 0, "assists": 3, "creepScore": 0}},
    {"summonerName": "B_Mid", "team": "ORDER", "position": "MIDDLE", "level": 5, "items": [{"itemID": 1052}], "scores": {"kills": 2, "deaths": 1, "assists": 1, "creepScore": 42}},
    {"summonerName": "B_Bot", "team": "ORDER", "position": "BOTTOM", "level": 4, "items": [{"itemID": 1055}], "scores": {"kills": 1, "deaths": 0, "assists": 1, "creepScore": 38}},
    {"summonerName": "B_Sup", "team": "ORDER", "position": "UTILITY", "level": 3, "items": [{"itemID": 1004}], "scores": {"kills": 0, "deaths": 1, "assists": 4, "creepScore": 0}},
    {"summonerName": "R_Top", "team": "CHAOS", "position": "TOP", "level": 4, "items": [{"itemID": 1036}], "scores": {"kills": 0, "deaths": 1, "assists": 0, "creepScore": 32}},
    {"summonerName": "R_Jgl", "team": "CHAOS", "position": "JUNGLE", "level": 4, "items": [], "scores": {"kills": 0, "deaths": 0, "assists": 1, "creepScore": 0}},
    {"summonerName": "R_Mid", "team": "CHAOS", "position": "MIDDLE", "level": 5, "items": [{"itemID": 1052}], "scores": {"kills": 1, "deaths": 2, "assists": 0, "creepScore": 45}},
    {"summonerName": "R_Bot", "team": "CHAOS", "position": "BOTTOM", "level": 4, "items": [], "scores": {"kills": 0, "deaths": 1, "assists": 0, "creepScore": 35}},
    {"summonerName": "R_Sup", "team": "CHAOS", "position": "UTILITY", "level": 3, "items": [{"itemID": 1004}], "scores": {"kills": 0, "deaths": 0, "assists": 1, "creepScore": 0}}
]

COMMON_ITEMS = [1001, 1036, 1052, 1053, 3070, 3111, 3158, 6632, 6653, 3089]
events = []
event_id = 1
dragon_count = 0

def reset_simulation():
    global tick, events, event_id, dragon_count, players
    tick = 0
    events = []
    event_id = 1
    dragon_count = 0
    for p in players:
        p['level'] = random.randint(3, 5)
        p['items'] = []
        p['scores'] = {'kills': 0, 'deaths': 0, 'assists': 0, 'creepScore': 0}

@app.route('/liveclientdata/allgamedata')
def get_data():
    global tick, event_id, dragon_count
    
    # Reset if max time reached (35 mins)
    if tick >= MAX_TICKS:
        reset_simulation()
    
    # Calculate game time in seconds
    game_time = START_OFFSET + (tick * 10)
    tick += 1
    
    # Update players each tick
    for p in players:
        p['scores']['creepScore'] += random.randint(0, 2)
        if p['level'] < 18:
            p['level'] = min(18, p['level'] + random.randint(0, 1))
        if random.random() < 0.1 and len(p['items']) < 5:
            p['items'].append({"itemID": random.choice(COMMON_ITEMS)})
    
    # Random kills (1% per tick per team)
    if random.random() < 0.01:
        killer = random.choice(players)
        victim = random.choice([x for x in players if x['team'] != killer['team']])
        killer['scores']['kills'] += 1
        victim['scores']['deaths'] += 1
        for ally in players:
            if ally['team'] == killer['team'] and ally != killer:
                ally['scores']['assists'] += random.randint(0, 1)
        events.append({"EventID": event_id, "EventName": "ChampionKill", "EventTime": game_time, "KillerName": killer['summonerName'], "VictimName": victim['summonerName']})
        event_id += 1
    
    # Objectives (dragons max 6, baron after 15 mins/900 secs)
    if tick > 60 and random.random() < 0.005:  # 0.5% chance per tick
        if dragon_count < 6:
            events.append({"EventID": event_id, "EventName": "DragonKill", "EventTime": game_time, "KillerName": random.choice(players)['summonerName']})
            dragon_count += 1
            event_id += 1
        elif game_time > 900 and random.random() < 0.3:  # Baron after 15 mins
            events.append({"EventID": event_id, "EventName": "BaronKill", "EventTime": game_time, "KillerName": random.choice(players)['summonerName']})
            event_id += 1
    
    # Turrets
    if random.random() < 0.003:
        events.append({"EventID": event_id, "EventName": "TurretKilled", "EventTime": game_time, "KillerName": random.choice(players)['summonerName']})
        event_id += 1
    
    return jsonify({
        "gameData": {"gameTime": game_time},
        "allPlayers": players,
        "events": {"Events": events}
    })

if __name__ == '__main__':
    app.run(port=2999, debug=False)