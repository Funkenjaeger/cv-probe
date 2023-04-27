import tomlkit


class Config:

    def __init__(self, filename):
        self._filename = filename
        with open(filename, 'r') as file:
            self._config = tomlkit.load(file)

    def get(self, header, field):
        return self._config.get(header, None).get(field, None)

    def set(self, header, field, val):
        self._config[header][field] = val
        with open(self._filename, 'w') as file:
            tomlkit.dump(self._config, file)
