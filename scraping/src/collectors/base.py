from abc import ABC, abstractmethod
import pandas as pd

class BaseCollector(ABC):
    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch data and return as a dataframe."""
        pass
