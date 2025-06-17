import logging
from typing import Dict, List, Optional, Any
from .base_sidekick import BaseSidekick

# --------------------
# Configure logging
# --------------------
logger = logging.getLogger("sidekick_registry")

class SidekickRegistry:
    """
    Central registry for all Sidekick instances and their versions.
    
    This class manages the registration, versioning, and retrieval of all
    10 Sidekicks across the Data, Ops, and Support crews. It handles:
    - Loading and initializing all Sidekick instances
    - Version management and resolution
    - Availability and capability tracking
    - Dynamic registration and updates
    
    Integration Points:
    - Used by: SidekickEngine for Sidekick resolution
    - Manages: All 10 Sidekick instances with their versions
    - Configuration: Loads from hardcoded registration (extensible to YAML)
    - Logging: Uses structured logging for all registry operations
    
    Architecture:
    Registry format: {
        "sidekick_name": {
            "v1.0": SidekickInstance,
            "v2.1": SidekickInstance,
            "latest": "v2.1"
        }
    }
    """
    
    def __init__(self):
        """
        Initialize the registry and load all Sidekick instances.
        
        This creates instances of all available Sidekicks and registers them
        with their default versions and configurations.
        """
        
        logger.info("Initializing Sidekick Registry...")
        
        # Registry storage: {sidekick_name: {version: instance}}
        self._sidekicks: Dict[str, Dict[str, BaseSidekick]] = {}
        
        # Metadata tracking
        self._sidekick_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load all available Sidekicks into the registry
        self._register_all_sidekicks()
        
        logger.info(f"Sidekick Registry initialized with {len(self._sidekicks)} Sidekicks")
    
    def _register_all_sidekicks(self):
        """
        Register all available Sidekicks with their initial versions.
        
        This method creates instances of each implemented Sidekick and registers them
        in the registry with their default configurations. Uses safe imports with
        fallback handling for missing implementations.
        """
        
        logger.info("Registering available Sidekicks...")
        
        # Data Crew Sidekicks - Safe imports with fallbacks
        try:
            from .sidekicks import FixxySidekick
            self._register_sidekick_instance(FixxySidekick(), "data")
            logger.info("Registered FixxySidekick successfully")
        except ImportError as e:
            logger.warning(f"FixxySidekick not available: {e}")
            self._register_placeholder_sidekick("fixxy", "data", "Data Cleanup Specialist")
        
        # Try to import other Data Crew Sidekicks
        try:
            from .sidekicks import FindySidekick
            self._register_sidekick_instance(FindySidekick(), "data")
            logger.info("Registered FindySidekick successfully")
        except (ImportError, AttributeError) as e:
            logger.warning(f"FindySidekick not available: {e}")
            self._register_placeholder_sidekick("findy", "data", "Pattern Detection Specialist")
        
        try:
            from .sidekicks import PredictySidekick
            self._register_sidekick_instance(PredictySidekick(), "data")
            logger.info("Registered PredictySidekick successfully")
        except (ImportError, AttributeError) as e:
            logger.warning(f"PredictySidekick not available: {e}")
            self._register_placeholder_sidekick("predicty", "data", "Forecasting Specialist")
        
        # Ops Crew Sidekicks - All placeholders for now
        self._register_placeholder_sidekick("planny", "ops", "Scheduling Specialist")
        self._register_placeholder_sidekick("syncie", "ops", "Workflow Specialist")
        self._register_placeholder_sidekick("watchy", "ops", "Monitoring Specialist")
        self._register_placeholder_sidekick("trackie", "ops", "Audit Trail Specialist")
        
        # Support Crew Sidekicks - All placeholders for now
        self._register_placeholder_sidekick("helpie", "support", "Support Specialist")
        self._register_placeholder_sidekick("coachy", "support", "Training Specialist")
        self._register_placeholder_sidekick("greetie", "support", "Onboarding Specialist")
        
        logger.info(f"Registered {len(self._sidekicks)} Sidekicks successfully")
        
        # Log which Sidekicks are available
        for name in self._sidekicks:
            metadata = self._sidekick_metadata[name]
            logger.info(f"Available: {name} v{metadata['latest_version']} ({metadata['crew']} crew)")
    
    def _register_placeholder_sidekick(self, name: str, crew: str, display_name: str):
        """
        Register a placeholder Sidekick when the real implementation is not available.
        
        This ensures the system can still function even if some Sidekicks are not
        fully implemented yet.
        """
        
        # Import the placeholder from sidekicks.__init__.py
        try:
            import importlib
            sidekicks_module = importlib.import_module('.sidekicks', package=__name__.rsplit('.', 1)[0])
            
            # Get the placeholder class by name
            placeholder_class_name = f"{name.capitalize()}Sidekick"
            if name == "findy":
                placeholder_class_name = "FindySidekick"
            elif name == "fixxy":
                placeholder_class_name = "FixxySidekick"
            elif name == "predicty":
                placeholder_class_name = "PredictySidekick"
            elif name == "planny":
                placeholder_class_name = "PlannySidekick"
            elif name == "syncie":
                placeholder_class_name = "SyncieSidekick"
            elif name == "watchy":
                placeholder_class_name = "WatchySidekick"
            elif name == "trackie":
                placeholder_class_name = "TrackieSidekick"
            elif name == "helpie":
                placeholder_class_name = "HelpieSidekick"
            elif name == "coachy":
                placeholder_class_name = "CoachySidekick"
            elif name == "greetie":
                placeholder_class_name = "GreetieSidekick"
            
            placeholder_class = getattr(sidekicks_module, placeholder_class_name, None)
            
            if placeholder_class:
                placeholder_instance = placeholder_class()
                self._register_sidekick_instance(placeholder_instance, crew)
                logger.info(f"Registered placeholder for {name}")
            else:
                logger.error(f"No placeholder found for {name}")
                
        except Exception as e:
            logger.error(f"Failed to register placeholder for {name}: {e}")
    
    def _register_sidekick_instance(self, sidekick: Any, crew: str):
        """
        Register a single Sidekick instance in the registry.
        
        This handles the registration of an individual Sidekick with
        version tracking and metadata storage.
        
        Args:
            sidekick: The Sidekick instance to register
            crew: The crew this Sidekick belongs to (data/ops/support)
        """
        
        try:
            name = sidekick.name.lower()
            version = sidekick.get_version() if hasattr(sidekick, 'get_version') else sidekick.version
            
            logger.info(f"Registering Sidekick '{name}' version '{version}' (crew: {crew})")
            
            # Initialize Sidekick entry if not exists
            if name not in self._sidekicks:
                self._sidekicks[name] = {}
                self._sidekick_metadata[name] = {
                    "crew": crew,
                    "display_name": getattr(sidekick, 'display_name', f"{name.capitalize()} Sidekick"),
                    "supported_tasks": sidekick.get_supported_tasks() if hasattr(sidekick, 'get_supported_tasks') else [],
                    "requires_llm": sidekick.requires_llm() if hasattr(sidekick, 'requires_llm') else getattr(sidekick, '_requires_llm', False),
                    "latest_version": version,
                    "all_versions": [],
                    "enabled": True
                }
            
            # Register this version
            self._sidekicks[name][version] = sidekick
            self._sidekick_metadata[name]["all_versions"].append(version)
            
            # Update latest version (simple string comparison for now)
            current_latest = self._sidekick_metadata[name]["latest_version"]
            if self._compare_versions(version, current_latest) > 0:
                self._sidekick_metadata[name]["latest_version"] = version
            
            logger.debug(f"Sidekick '{name}' v{version} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register Sidekick: {e}")
    
    def get_sidekick(self, name: str, version: Optional[str] = None) -> BaseSidekick:
        """
        Retrieve a Sidekick instance from the registry.
        
        This is the main method used by the engine to get Sidekick instances
        for prompt generation. It handles version resolution and fallbacks.
        
        Args:
            name: Sidekick name (case-insensitive)
            version: Specific version requested (defaults to latest)
            
        Returns:
            BaseSidekick: The requested Sidekick instance
            
        Raises:
            KeyError: If Sidekick or version not found
        """
        
        name = name.lower().strip()
        
        # Check if Sidekick exists
        if name not in self._sidekicks:
            available = ", ".join(self.get_all_sidekick_names())
            raise KeyError(f"Sidekick '{name}' not found. Available: {available}")
        
        # Determine version to use
        if version is None:
            version = self._sidekick_metadata[name]["latest_version"]
            logger.debug(f"Using latest version '{version}' for Sidekick '{name}'")
        
        # Check if version exists
        if version not in self._sidekicks[name]:
            available_versions = ", ".join(self._sidekicks[name].keys())
            raise KeyError(f"Version '{version}' not found for Sidekick '{name}'. "
                          f"Available versions: {available_versions}")
        
        sidekick_instance = self._sidekicks[name][version]
        logger.debug(f"Retrieved Sidekick '{name}' version '{version}'")
        
        return sidekick_instance
    
    def has_sidekick(self, name: str) -> bool:
        """
        Check if a Sidekick is registered in the registry.
        
        Args:
            name: Sidekick name to check
            
        Returns:
            bool: True if Sidekick exists
        """
        
        return name.lower().strip() in self._sidekicks
    
    def get_all_sidekick_names(self) -> List[str]:
        """
        Get a list of all registered Sidekick names.
        
        Returns:
            List[str]: All registered Sidekick names
        """
        
        return list(self._sidekicks.keys())
    
    def get_all_sidekicks(self) -> List[BaseSidekick]:
        """
        Get all registered Sidekick instances (latest versions only).
        
        Returns:
            List[BaseSidekick]: Latest version of each registered Sidekick
        """
        
        sidekicks = []
        for name in self._sidekicks:
            try:
                latest_sidekick = self.get_sidekick(name)  # Gets latest version
                sidekicks.append(latest_sidekick)
            except Exception as e:
                logger.warning(f"Could not retrieve latest version of Sidekick '{name}': {e}")
        
        return sidekicks
    
    def get_sidekick_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific Sidekick.
        
        Args:
            name: Sidekick name
            
        Returns:
            Optional[Dict]: Sidekick metadata or None if not found
        """
        
        name = name.lower().strip()
        return self._sidekick_metadata.get(name)
    
    def get_sidekicks_by_crew(self, crew: str) -> List[BaseSidekick]:
        """
        Get all Sidekicks belonging to a specific crew.
        
        Args:
            crew: Crew name (data/ops/support)
            
        Returns:
            List[BaseSidekick]: Sidekicks in the specified crew
        """
        
        crew_sidekicks = []
        for name, metadata in self._sidekick_metadata.items():
            if metadata["crew"] == crew.lower():
                try:
                    sidekick = self.get_sidekick(name)
                    crew_sidekicks.append(sidekick)
                except Exception as e:
                    logger.warning(f"Could not retrieve Sidekick '{name}': {e}")
        
        return crew_sidekicks
    
    def get_latest_enabled_version(self, name: str) -> Optional[BaseSidekick]:
        """
        Get the latest enabled version of a Sidekick.
        
        This is used as a fallback when a specific version is disabled
        but other versions might be available.
        
        Args:
            name: Sidekick name
            
        Returns:
            Optional[BaseSidekick]: Latest enabled version or None
        """
        
        name = name.lower().strip()
        
        if name not in self._sidekicks:
            return None
        
        # Check if the Sidekick is globally enabled
        if not self._sidekick_metadata[name].get("enabled", True):
            return None
        
        # Get latest version (they're all enabled for now in this simple implementation)
        try:
            return self.get_sidekick(name)
        except Exception:
            return None
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.
        
        Simple version comparison for now. Could be enhanced with
        semantic versioning library if needed.
        
        Args:
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            int: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        
        try:
            # Simple numeric comparison for versions like "1.0", "2.1"
            v1_parts = [int(x) for x in version1.replace('v', '').split('.')]
            v2_parts = [int(x) for x in version2.replace('v', '').split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            # Compare part by part
            for i in range(max_len):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            return 0  # Equal
            
        except Exception:
            # Fallback to string comparison
            return (version1 > version2) - (version1 < version2)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current registry state.
        
        Useful for debugging and monitoring the registry health.
        
        Returns:
            Dict: Registry summary with counts and status information
        """
        
        summary = {
            "total_sidekicks": len(self._sidekicks),
            "crews": {
                "data": len(self.get_sidekicks_by_crew("data")),
                "ops": len(self.get_sidekicks_by_crew("ops")),
                "support": len(self.get_sidekicks_by_crew("support"))
            },
            "sidekicks": {}
        }
        
        # Add details for each Sidekick
        for name in self._sidekicks:
            metadata = self._sidekick_metadata[name]
            summary["sidekicks"][name] = {
                "crew": metadata["crew"],
                "latest_version": metadata["latest_version"],
                "total_versions": len(metadata["all_versions"]),
                "enabled": metadata["enabled"],
                "supported_tasks": len(metadata["supported_tasks"]),
                "requires_llm": metadata["requires_llm"]
            }
        
        return summary