import sys
sys.path.append('C:/Users/nsmal/league5min')
from app import app

with app.test_client() as c:
    # Test start simulator
    r = c.post('/api/start_simulator')
    print('Start sim:', r.get_json())
    
    # Test live prediction
    r = c.get('/api/live_prediction')
    print('Status:', r.status_code)
    data = r.get_json()
    if r.status_code == 200:
        print('OK - blue:', data.get('blue_prob'))
    else:
        print('ERROR:', data)