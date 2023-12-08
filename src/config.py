import yaml


class Config:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def get(self, key):
        value = None
        current_value = self.config
        for k in key.split('.'):
            current_value = current_value[k]
        return current_value


config = Config('config.yaml')
