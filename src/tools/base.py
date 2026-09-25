from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    """Abstract base class for all Agent Tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, query_or_input: str, **kwargs) -> Dict[str, Any]:
        """Execute tool and return structured dictionary response."""
        pass
