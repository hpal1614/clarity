"""
Base Sidekick Abstract Class

This module defines the abstract base class that all Sidekicks must inherit from.
It provides the standard interface and common functionality that ensures
consistent behavior across all 10 Sidekicks in the Clarity platform.

Each Sidekick specializes in a specific type of prompt generation, but all
follow the same patterns for initialization, task handling, and LLM integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from langchain.prompts import PromptTemplate
import logging


logger = logging.getLogger(__name__)

class BaseSidekick(ABC):
    """
    Abstract base class for all Sidekick implementations.
    
    This class defines the interface that all Sidekicks must implement
    to ensure consistent behavior and integration with the Sidekick Engine.
    """
    
    def __init__(self, name: str, version: str, display_name: str):
        """
        Initialize the base Sidekick with common attributes.
        
        Args:
            name: Unique identifier for the Sidekick (e.g., "fixxy")
            version: Version string (e.g., "v1.0")
            display_name: Human-readable name (e.g., "Fixxy - Data Cleanup Specialist")
        """
        
        self.name = name.lower()
        self.version = version
        self.display_name = display_name
        
        # Common attributes
        self._enabled = True
        self._requires_llm = False  # Subclasses can override
        self._supported_tasks = []  # Subclasses must populate
        
        # Statistics tracking
        self._stats = {
            "requests_handled": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "llm_calls_made": 0,
            "cache_hits": 0
        }
        
        # Initialize LangChain components
        try:
            self._initialize_langchain_components()
        except Exception as e:
            logger.warning(f"LangChain initialization failed for {self.name}: {e}")
        
        logger.info(f"Initialized {self.display_name} version {self.version}")
    
    @abstractmethod
    def _initialize_langchain_components(self):
        """
        Initialize LangChain-specific components for this Sidekick.
        
        This method should set up:
        - Prompt templates
        - LLM chains
        - Output parsers
        - Any other LangChain components
        
        Must be implemented by each Sidekick.
        """
        pass
    
    @abstractmethod
    async def generate_prompt_template(self, task_type: str, context: Dict[str, Any], 
                                     langchain_manager: Any, job_id: str):
        """
        Generate a prompt template for the given task and context.
        
        This is the main method called by the Sidekick Engine.
        
        Args:
            task_type: Type of task to generate prompt for
            context: Job context and parameters
            langchain_manager: LangChain manager for LLM calls
            job_id: Job ID for logging and traceability
            
        Returns:
            GeneratedPromptTemplate: The generated prompt template
        """
        pass
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task types"""
        return self._supported_tasks.copy()
    
    def supports_task(self, task_type: str) -> bool:
        """Check if this Sidekick supports a specific task type"""
        return task_type.lower() in [t.lower() for t in self._supported_tasks]
    
    def requires_llm(self, task_type: str = None) -> bool:
        """Check if this Sidekick requires LLM assistance"""
        return self._requires_llm
    
    def is_enabled(self) -> bool:
        """Check if this Sidekick is currently enabled"""
        return self._enabled
    
    def enable(self):
        """Enable this Sidekick"""
        self._enabled = True
        logger.info(f"Enabled {self.name}")
    
    def disable(self):
        """Disable this Sidekick"""
        self._enabled = False
        logger.info(f"Disabled {self.name}")
    
    def get_version(self) -> str:
        """Get Sidekick version"""
        return self.version
    
    def get_name(self) -> str:
        """Get Sidekick name"""
        return self.name
    
    def get_display_name(self) -> str:
        """Get human-readable display name"""
        return self.display_name
    
    def increment_stat(self, stat_name: str, amount: int = 1):
        """Increment a statistic counter"""
        if stat_name in self._stats:
            self._stats[stat_name] += amount
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics for this Sidekick"""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset all statistics to zero"""
        for key in self._stats:
            self._stats[key] = 0
        logger.info(f"Reset statistics for {self.name}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get Sidekick metadata"""
        return {
            "name": self.name,
            "version": self.version,
            "display_name": self.display_name,
            "enabled": self._enabled,
            "requires_llm": self._requires_llm,
            "supported_tasks": self._supported_tasks,
            "stats": self._stats
        }
    
    def __str__(self) -> str:
        """String representation of the Sidekick"""
        return f"{self.display_name} ({self.version})"
    
    def __repr__(self) -> str:
        """Developer representation of the Sidekick"""
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"