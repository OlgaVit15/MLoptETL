from abc import ABC, abstractmethod


class BaseCollector(ABC):
    @abstractmethod
    def get_current_state(self) -> dict: pass


class BaseBalancer(ABC):
    @abstractmethod
    def balance(self, ml_output: dict, cluster_state: dict) -> dict: pass


class FeatureHasher(ABC):
    @abstractmethod
    def compute_hash(self, ) -> dict: pass
