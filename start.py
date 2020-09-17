import json
from concurrent.futures import ProcessPoolExecutor
import sys
from app import ModelRunner

def run_model(config):
    if config["run_this"]:
        model_runner = ModelRunner(config)
        model_runner.run()

model_configs = json.load( open("configs.json") )

for i, config in enumerate(model_configs["configs"]):
    run_model(config)

# with ProcessPoolExecutor() as executor:
#     executor.map(run_model, model_configs["configs"])