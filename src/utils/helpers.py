import yaml

ROOT = "C:/Users/Ameet/Documents/research/repositories/chinook_for_fetch/"
# ROOT = 

with open(ROOT + 'config.yaml', 'r') as file:
    config = yaml.safe_load(file)