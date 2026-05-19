"""
gpulens.collectors.base
=======================
Abstract base class all collectors must implement.
"""
from abc import ABC, abstractmethod
from gpulens.models.cluster import ClusterSnapshot


class BaseCollector(ABC):

    @abstractmethod
    def collect(self) -> ClusterSnapshot:
        """
        Pull metrics and return a ClusterSnapshot.
        Should be a point-in-time snapshot, not a streaming call.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Lightweight connectivity check. Returns True if the data source
        is reachable. Should not raise; return False on any failure.
        """
        ...
