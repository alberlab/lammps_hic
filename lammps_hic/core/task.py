

class MapReduceJob(object):
    def __init__(self, task, args, controller):
        self.task = task
        self.args = args
        self.controller = controller
        self.results = None

    def run(self):
        self.results = self.controller.map(self.task, self.args)

    def results(self):
        return self.results