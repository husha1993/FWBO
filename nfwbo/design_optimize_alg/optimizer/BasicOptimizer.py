import abc


class BasicOptimizer(object):
    def __init__(self):
        self.X = []
        self.Y = []

    @abc.abstractmethod
    def configure(self, **kwargs):
        '''
        set the hyper-parameters of the optimizer from config
        '''
        return

    @abc.abstractmethod
    def optimize(self):
        return

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        return

    @abc.abstractmethod
    def record(self, **kwargs):
        return







