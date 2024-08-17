from abc import ABC, abstractmethod

class AbstractConfig(ABC):
    @abstractmethod
    def __init__(self):
        pass