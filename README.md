# Bachelor-Thesis-Experiment

Code to run Experiment on lab. This documentation is for linux systems.

# Setup
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run experiment
to run everything at once
```
python experiment/run_experiment.py
```

to run separately
```
python experiment/run_experiment_matching.py
python experiment/run_experiment_likert.py
```

# Run analysis
Make sure you are in the right directory first.
```
cd analysis
python run_analysis.py
```
