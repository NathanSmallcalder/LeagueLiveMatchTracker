1. Backend - Add Simulator Spawner to app.py

Add a new API endpoint /api/start_simulator
Use Python's subprocess.Popen to launch simulator.py in the background (detached process)
Returns status to confirm the simulator is running
2. Frontend - Update templates/index.html

Add a "Run Demo" button in the header that calls /api/start_simulator
Add an "Info" button that opens a modal popup with instructions:
How to run the app normally
How to run the demo
How to access the browser interface