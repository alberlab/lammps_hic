
class Step(object):
    def __init__(self, config):
        pass

    def setup(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def cleanup(self):
        raise NotImplementedError()