import os

import yaml


class Config:
    def __init__(self, config_file_path):
        self.load(config_file_path)

    def load(self, config_file_path):
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def __getitem__(self, key):
        current_value = self.config
        for k in key.split('.'):
            current_value = current_value[k]
        return current_value

    def __str__(self):
        return str(self.config)
        

_config_file_path = os.getenv('NDA_CONFIG', 'configs/main.yaml')
config = Config(_config_file_path)

