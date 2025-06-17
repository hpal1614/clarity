import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime
import threading

from .models import (
    SidekickEngineRequest,
    SidekickEngineResponse, 
    GeneratedPromptTemplate,
    PromptTemplateMetadata
)
from .sidekick_registry import SidekickRegistry
from .base_sidekick import BaseSidekick
from .langchain_integration import LangChainManager

# --------------------
# Configure logging
# --------------------
logger = logging.getLogger("sidekick_engine_core")

class SidekickEngine:
    """
    Core engine that manages all Sidekick instances and routes requests.
    
    This class acts as the central registry and orchestrator for all 10 Sidekicks
    across the Data, Ops, and Support crews. It handles version management,
    configuration loading, and provides a unified interface for prompt generation.
    
    Integration Points:
    - Input: Receives requests from Sidekick Router via main.py
    - Output: Returns structured prompt templates
    - External: May call Relayer Gateway for LLM assistance
    - Registry: Manages all Sidekick instances and their versions
    - LangChain: Integrates with LangChain for prompt generation
    
    Architecture:
    - Registry stores all available Sidekicks by name and version
    - LangChain Manager handles all LLM interactions
    - Base Sidekick interface ensures consistent behavior
    - Error handling with fallbacks and graceful degradation
    """
    
    def __init__(self):
        """
        Initialize the Sidekick Engine with all components.
        
        This loads all 10 Sidekick definitions, sets up LangChain integration,
        and prepares the engine for processing requests.
        """
        
        logger.info("Initializing Sidekick Engine...")
        
        # Initialize the Sidekick registry
        # This loads all 10 Sidekicks from configuration
        self.registry = SidekickRegistry()
        logger.info(f"Sidekick Registry initialized with {len(self.registry.get_all_sidekicks())} Sidekicks")
        
        # Initialize LangChain integration
        # This sets up the LLM connection and chain management
        self.langchain_manager = LangChainManager()
        logger.info("LangChain Manager initialized")
        
        # Engine configuration and state
        self.engine_version = "v1.0"
        self.stats = {
            "requests_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "llm_calls_made": 0,
            "fallbacks_used": 0
        }
        self._stats_lock = threading.Lock()
        
        logger.info("Sidekick Engine initialization complete")
    
    def increment_stat(self, key):
        with self._stats_lock:
            self.stats[key] += 1
    
    def process_request(self, request: SidekickEngineRequest) -> SidekickEngineResponse:
        """
        Main request processing method - routes to appropriate Sidekick.
        
        This is the primary entry point called by the Azure Function.
        It handles the complete flow: validation → routing → execution → response.
        
        Args:
            request: Validated request containing sidekick_name, task_type, context
            
        Returns:
            SidekickEngineResponse: Contains generated prompt template and metadata
            
        Integration Note:
            Response format must match what Sidekick Router expects for seamless handoff
        """
        
        # Increment request counter for monitoring
        self.increment_stat("requests_processed")
        
        # Extract key information for logging (matches manager's [job_id] pattern)
        job_id = request.job_id
        sidekick_name = request.sidekick_name.lower().strip()
        task_type = request.task_type
        
        logger.info(f"[{job_id}] Processing request for Sidekick '{sidekick_name}' task '{task_type}'")
        
        # Record start time for performance tracking
        start_time = time.time()
        
        try:
            # Step 1: Validate request format and required fields
            # Ensures all necessary data is present before processing
            validation_result = self._validate_request(request)
            if not validation_result["valid"]:
                logger.error(f"[{job_id}] Request validation failed: {validation_result['error']}")
                return self._create_error_response(
                    "VALIDATION_ERROR", 
                    validation_result["error"], 
                    request
                )
            
            # Step 2: Resolve Sidekick instance from registry
            # Handles version resolution, fallbacks, and availability checks
            try:
                sidekick_instance = self._resolve_sidekick(
                    sidekick_name, 
                    request.requested_version, 
                    job_id
                )
                
                logger.info(f"[{job_id}] Resolved Sidekick '{sidekick_name}' "
                           f"version '{sidekick_instance.get_version()}'")
                
            except Exception as e:
                logger.error(f"[{job_id}] Sidekick resolution failed: {str(e)}")
                return self._create_error_response(
                    "SIDEKICK_NOT_FOUND", 
                    f"Could not resolve Sidekick '{sidekick_name}': {str(e)}", 
                    request
                )
            
            # Step 3: Check if Sidekick can handle the requested task
            if not sidekick_instance.can_handle_task(task_type):
                supported_tasks = ", ".join(sidekick_instance.get_supported_tasks())
                error_msg = (f"Sidekick '{sidekick_name}' does not support task '{task_type}'. "
                           f"Supported tasks: {supported_tasks}")
                logger.error(f"[{job_id}] {error_msg}")
                return self._create_error_response(
                    "UNSUPPORTED_TASK", 
                    error_msg, 
                    request
                )
            
            # Step 4: Generate prompt template using the Sidekick
            # This is where the actual prompt generation happens
            try:
                logger.info(f"[{job_id}] Generating prompt template...")
                
                generation_start = time.time()
                
                # Call the Sidekick's generation method
                generation_result = sidekick_instance.generate_prompt(
                    task_type=task_type,
                    context=request.job_context,
                    langchain_manager=self.langchain_manager,
                    job_id=job_id
                )
                
                generation_time = int((time.time() - generation_start) * 1000)
                
                # Extract prompt information from generation result
                prompt_text = generation_result.get("prompt_text", "")
                if not prompt_text:
                    raise Exception("No prompt text generated")
                
                # Determine input variables from the prompt template
                input_variables = generation_result.get("input_variables", [])
                if not input_variables:
                    # Auto-detect variables from template (simple {var} pattern)
                    import re
                    input_variables = list(set(re.findall(r'\{(\w+)\}', prompt_text)))
                
                # Create the structured prompt template
                prompt_template = GeneratedPromptTemplate(
                    template_content=prompt_text,
                    input_variables=input_variables,
                    template_format="langchain",
                    expected_output=generation_result.get("expected_output", "structured_text")
                )
                
                # Create metadata for tracking and debugging
                template_metadata = PromptTemplateMetadata(
                    sidekick_used=sidekick_name,
                    sidekick_version=sidekick_instance.get_version(),
                    generation_method=generation_result.get("method", "template"),
                    llm_used=generation_result.get("llm_used", False),
                    model_preference=generation_result.get("model_preference"),
                    max_tokens=generation_result.get("max_tokens"),
                    temperature=generation_result.get("temperature"),
                    metadata=generation_result.get("metadata", {})
                )
                
                # Create the final response
                response = SidekickEngineResponse(
                    request_id=request.request_id,
                    job_id=job_id,
                    status="success",
                    prompt_template=prompt_template,
                    processed_at=datetime.utcnow().isoformat(),
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    engine_version=self.engine_version,
                    fallback_used=generation_result.get("fallback_method") if generation_result.get("method") == "fallback" else None,
                    metadata=template_metadata
                )
                
                # Update success statistics
                self.increment_stat("successful_generations")
                if generation_result.get("llm_used", False):
                    self.increment_stat("llm_calls_made")
                if generation_result.get("method") == "fallback":
                    self.increment_stat("fallbacks_used")
                
                logger.info(f"[{job_id}] Request processed successfully - returning prompt template")
                return response
                
            except Exception as e:
                logger.error(f"[{job_id}] Failed to generate prompt: {str(e)}", exc_info=True)
                
                # Try fallback generation if primary failed
                try:
                    logger.info(f"[{job_id}] Attempting fallback prompt generation...")
                    
                    fallback_result = sidekick_instance.generate_fallback_prompt(
                        task_type=task_type,
                        context=request.job_context
                    )
                    
                    if fallback_result:
                        self.increment_stat("fallbacks_used")
                        
                        # Create response with fallback prompt
                        prompt_template = GeneratedPromptTemplate(
                            template_content=fallback_result.get("prompt_text", ""),
                            input_variables=fallback_result.get("input_variables", ["data_context"]),
                            template_format="langchain",
                            expected_output="text"
                        )
                        
                        template_metadata = PromptTemplateMetadata(
                            sidekick_used=sidekick_name,
                            sidekick_version=sidekick_instance.get_version(),
                            generation_method="fallback",
                            llm_used=False
                        )
                        
                        response = SidekickEngineResponse(
                            request_id=request.request_id,
                            job_id=job_id,
                            status="fallback",
                            prompt_template=prompt_template,
                            processed_at=datetime.utcnow().isoformat(),
                            processing_time_ms=int((time.time() - start_time) * 1000),
                            engine_version=self.engine_version,
                            fallback_used="template_fallback",
                            metadata=template_metadata
                        )
                        
                        logger.info(f"[{job_id}] Fallback generation successful")
                        self.increment_stat("successful_generations")
                        return response
                        
                except Exception as fallback_error:
                    logger.error(f"[{job_id}] Fallback generation also failed: {str(fallback_error)}")
                
                # Both primary and fallback failed
                self.increment_stat("failed_generations")
                return self._create_error_response(
                    "GENERATION_FAILED", 
                    f"Failed to generate prompt: {str(e)}", 
                    request
                )
        
        except Exception as e:
            # Catch-all error handler for unexpected failures
            logger.error(f"[{job_id}] Unexpected error in engine processing: {str(e)}", exc_info=True)
            self.increment_stat("failed_generations")
            return self._create_error_response(
                "INTERNAL_ERROR", 
                f"Internal engine error: {str(e)}", 
                request
            )
    
    def _validate_request(self, request: SidekickEngineRequest) -> Dict[str, Any]:
        """
        Validate incoming request for completeness and correctness.
        
        Checks that all required fields are present and that the request
        is well-formed for processing by the engine.
        
        Args:
            request: The incoming request to validate
            
        Returns:
            Dict containing validation result and any error messages
        """
        
        try:
            # Check required fields
            if not request.job_id:
                return {"valid": False, "error": "Missing required field: job_id"}
            
            if not request.sidekick_name:
                return {"valid": False, "error": "Missing required field: sidekick_name"}
            
            if not request.task_type:
                return {"valid": False, "error": "Missing required field: task_type"}
            
            # Validate Sidekick name format
            sidekick_name = request.sidekick_name.lower().strip()
            if not sidekick_name.replace("_", "").isalnum():
                return {"valid": False, "error": f"Invalid Sidekick name format: {request.sidekick_name}"}
            
            # Check if Sidekick exists in registry
            if not self.registry.has_sidekick(sidekick_name):
                available_sidekicks = ", ".join(self.registry.get_all_sidekick_names())
                return {
                    "valid": False, 
                    "error": f"Unknown Sidekick '{sidekick_name}'. Available: {available_sidekicks}"
                }
            
            # Validate job context is not None
            if request.job_context is None:
                return {"valid": False, "error": "job_context cannot be None"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _resolve_sidekick(self, sidekick_name: str, requested_version: Optional[str], 
                         job_id: str) -> BaseSidekick:
        """
        Resolve a Sidekick instance from the registry.
        
        Handles version resolution, availability checks, and fallback logic
        to ensure we always get a usable Sidekick instance.
        
        Args:
            sidekick_name: Name of the Sidekick to resolve
            requested_version: Specific version requested (optional)
            job_id: Job ID for logging
            
        Returns:
            BaseSidekick: The resolved Sidekick instance
            
        Raises:
            Exception: If Sidekick cannot be resolved
        """
        
        logger.debug(f"[{job_id}] Resolving Sidekick '{sidekick_name}' version '{requested_version or 'latest'}'")
        
        try:
            # Get Sidekick from registry
            sidekick = self.registry.get_sidekick(sidekick_name, requested_version)
            
            # Verify the Sidekick is enabled
            if hasattr(sidekick, 'is_enabled') and not sidekick.is_enabled():
                # Try to get the latest enabled version
                latest_enabled = self.registry.get_latest_enabled_version(sidekick_name)
                if latest_enabled:
                    logger.warning(f"[{job_id}] Requested version disabled, using latest enabled version")
                    return latest_enabled
                else:
                    raise Exception(f"Sidekick '{sidekick_name}' is disabled with no enabled versions")
            
            return sidekick
            
        except KeyError as e:
            # Sidekick or version not found
            raise Exception(f"Sidekick not found: {str(e)}")
        except Exception as e:
            # Other resolution errors
            raise Exception(f"Failed to resolve Sidekick: {str(e)}")
    
    def _create_error_response(self, error_code: str, error_message: str, 
                             request: SidekickEngineRequest) -> SidekickEngineResponse:
        """
        Create a standardized error response.
        
        Follows the manager's error response pattern to ensure consistency
        across all Clarity components.
        
        Args:
            error_code: Standardized error code
            error_message: Human-readable error description
            request: Original request for context
            
        Returns:
            SidekickEngineResponse: Error response in standard format
        """
        
        return SidekickEngineResponse(
            request_id=request.request_id,
            job_id=request.job_id,
            status="error",
            prompt_template=None,
            error_code=error_code,
            error_message=error_message,
            processed_at=datetime.utcnow().isoformat(),
            processing_time_ms=0,
            engine_version=self.engine_version
        )
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get current engine statistics for monitoring and debugging.
        
        Returns:
            Dict containing engine performance and usage statistics
        """
        
        return {
            "engine_version": self.engine_version,
            "stats": self.stats.copy(),
            "registry_info": {
                "total_sidekicks": len(self.registry.get_all_sidekicks()),
                "enabled_sidekicks": len([s for s in self.registry.get_all_sidekicks() if hasattr(s, 'is_enabled') and s.is_enabled()]),
                "available_sidekick_names": self.registry.get_all_sidekick_names()
            },
            "langchain_status": self.langchain_manager.get_status() if hasattr(self.langchain_manager, 'get_status') else "active"
        }