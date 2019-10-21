import configparser


class ConfigHelper:
    def __init__(self, config: str = 'config.ini'):
        parser = configparser.ConfigParser()
        parser.read(config)
        self.config = []
        sections = parser.sections()
        for s in range(len(sections)):
            dictionary = dict(data_path=parser.get(sections[s], 'data_path'),
                              number_of_steps=parser.getint(sections[s], 'number_of_steps'),
                              batch_size=parser.getint(sections[s], 'batch_size'),
                              hidden_size=parser.getint(sections[s], 'hidden_size'),
                              number_of_epochs=parser.getint(sections[s], 'number_of_epochs'),
                              seed=parser.getint(sections[s], 'seed'),
                              temperature=parser.getfloat(sections[s], 'temperature'),
                              number_of_predictions=parser.getint(sections[s], 'number_of_predictions'))
            self.config.append(dictionary)
