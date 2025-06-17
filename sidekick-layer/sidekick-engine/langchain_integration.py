import logging
import os
import httpx
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from .clarity_secrets import get_secret

# --------------------
# Configure logging
# --------------------
logger = logging.getLogger("langchain_integration")

class RelayerLLM(LLM):
    """
    Custom LangChain LLM that calls the Relayer Gateway.
    
    This class implements the LangChain LLM interface to seamlessly 
    integrate with existing LangChain chains and prompts while routing 
    all LLM calls through the Clarity Relayer Gateway.
    
    Integration Points:
    - Implements: LangChain LLM interface
    - Calls: Relayer Gateway via HTTP (matches manager's pattern)
    - Used by: All Sidekicks for LLM-assisted prompt generation
    - Authentication: Uses manager's token pattern
    
    This enables Sidekicks to use standard LangChain patterns while
    ensuring all LLM calls go through the secure Relayer infrastructure.
    """
    
    # LLM configuration attributes
    relayer_gateway_url: str
    relayer_gateway_token: str
    timeout_sec: int
    model_name: str = "gpt-4"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    def __init__(self, **kwargs):
        """
        Initialize the Relayer LLM with connection settings.
        
        Loads configuration and credentials needed to call the
        Relayer Gateway following the manager's established patterns.
        """
        
        super().__init__(**kwargs)
        
        # Load Relayer Gateway connection settings
        try:
            self.relayer_gateway_url = os.getenv("RELAYER_SIDEKICK_GATEWAY_URL")
            if not self.relayer_gateway_url:
                raise ValueError("RELAYER_SIDEKICK_GATEWAY_URL environment variable not set")
                
            self.relayer_gateway_token = get_secret("sidekick-layer--relayer-gateway-token")
            self.timeout_sec = int(os.getenv("RELAYER_TIMEOUT_SEC", "30"))
            
            logger.info("RelayerLLM initialized with Relayer Gateway connection")
            
        except Exception as e:
            logger.error(f"Failed to initialize RelayerLLM: {e}")
            raise
        
        # Statistics tracking
        self.stats = {
            "calls_made": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens_used": 0
        }
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type for LangChain compatibility."""
        return "relayer_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """
        Main LangChain interface method for LLM calls.
        
        This method is called by LangChain chains and implements the
        core LLM calling logic routed through the Relayer Gateway.
        
        Args:
            prompt: The prompt text to send to the LLM
            stop: Stop sequences (optional)
            run_manager: LangChain callback manager
            **kwargs: Additional parameters
            
        Returns:
            str: LLM response text
            
        Integration Note:
            This method follows the manager's HTTP calling pattern to
            ensure consistency with other gateway communications.
        """
        
        self.stats["calls_made"] += 1
        
        try:
            # Extract additional parameters from kwargs
            job_id = kwargs.get("job_id", "langchain_call")
            task_type = kwargs.get("task_type", "general")
            sidekick_name = kwargs.get("sidekick_name", "unknown")
            
            logger.info(f"[{job_id}] RelayerLLM making call for Sidekick '{sidekick_name}' task '{task_type}'")
            
            # Prepare the request payload (matches manager's SidekickLLMRequest format)
            request_payload = {
                "llm_request": {
                    "request_id": str(uuid.uuid4()),
                    "job_id": job_id,
                    "prompt_text": prompt,
                    "model_preference": kwargs.get("model_preference", self.model_name),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "task_type": task_type,
                    "sidekick_name": sidekick_name,
                    "context_metadata": kwargs.get("context_metadata", {}),
                    "stop_sequences": stop
                },
                "caller_component": sidekick_name,
                "timestamp_requested": datetime.utcnow().isoformat()
            }
            
            # Make the HTTP call to Relayer Gateway (matches manager's pattern)
            with httpx.Client(timeout=self.timeout_sec) as client:
                response = client.post(
                    url=self.relayer_gateway_url,
                    json=request_payload,
                    headers={
                        "x-internal-token": self.relayer_gateway_token,
                        "Content-Type": "application/json"
                    }
                )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract LLM output from response
            llm_output = self._extract_llm_output(response_data)
            
            # Update statistics
            self.stats["successful_calls"] += 1
            if "token_count_response" in response_data:
                self.stats["total_tokens_used"] += response_data.get("token_count_response", 0)
            
            logger.info(f"[{job_id}] RelayerLLM call successful")
            return llm_output
            
        except httpx.TimeoutException:
            self.stats["failed_calls"] += 1
            error_msg = f"RelayerLLM call timed out after {self.timeout_sec} seconds"
            logger.error(f"[{job_id}] {error_msg}")
            raise Exception(error_msg)
            
        except httpx.HTTPStatusError as e:
            self.stats["failed_calls"] += 1
            error_msg = f"RelayerLLM HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{job_id}] {error_msg}")
            raise Exception(error_msg)
            
        except Exception as e:
            self.stats["failed_calls"] += 1
            error_msg = f"RelayerLLM call failed: {str(e)}"
            logger.error(f"[{job_id}] {error_msg}", exc_info=True)
            raise Exception(error_msg)
    
    def _extract_llm_output(self, response_data: Dict[str, Any]) -> str:
        """
        Extract LLM output from Relayer Gateway response.
        
        Args:
            response_data: Response JSON from Relayer Gateway
            
        Returns:
            str: Extracted LLM output text
        """
        
        # Handle different response formats
        if "llm_response" in response_data:
            return response_data["llm_response"].get("llm_output", "")
        elif "llm_output" in response_data:
            return response_data["llm_output"]
        elif "response" in response_data:
            return response_data["response"]
        else:
            logger.warning("No LLM output found in response, returning empty string")
            return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for this LLM instance.
        
        Returns:
            Dict[str, Any]: Performance and usage statistics
        """
        
        return self.stats.copy()


class LangChainManager:
    """
    Manager class for LangChain integration within the Sidekick Engine.
    
    This class provides a unified interface for Sidekicks to interact with
    LangChain components and the Relayer Gateway. It manages LLM instances,
    handles configuration, and provides convenience methods for common operations.
    
    Integration Points:
    - Used by: All Sidekicks for LLM interactions
    - Manages: RelayerLLM instances and LangChain chains
    - Provides: Simplified interface for prompt execution
    - Coordinates: Between Sidekicks and Relayer Gateway
    """
    
    def __init__(self):
        """
        Initialize the LangChain Manager with default configuration.
        
        Sets up the RelayerLLM instance and prepares the manager
        for use by Sidekicks.
        """
        
        logger.info("Initializing LangChain Manager...")
        
        # Initialize the custom Relayer LLM
        try:
            self.relayer_llm = RelayerLLM()
            logger.info("RelayerLLM instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create RelayerLLM: {e}")
            raise
        
        # Manager configuration
        self.default_model = "gpt-4"
        self.default_max_tokens = 1000
        self.default_temperature = 0.7
        
        # Statistics tracking
        self.stats = {
            "manager_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0
        }
        
        logger.info("LangChain Manager initialized successfully")
    
    def get_llm(self) -> RelayerLLM:
        """
        Get the RelayerLLM instance for direct use.
        
        Returns:
            RelayerLLM: The configured LLM instance
        """
        return self.relayer_llm
    
    def call_llm(self, prompt_text: str, task_type: str, sidekick_name: str, 
                 job_id: str, **kwargs) -> Dict[str, Any]:
        """
        Make an LLM call via the Relayer Gateway.
        
        This is the main method used by Sidekicks to request LLM assistance.
        It handles the call coordination and returns structured results.
        
        Args:
            prompt_text: The prompt to send to the LLM
            task_type: Type of task being performed
            sidekick_name: Name of the calling Sidekick
            job_id: Job ID for logging and traceability
            **kwargs: Additional parameters for the LLM call
            
        Returns:
            Dict[str, Any]: LLM response with metadata
            
        Integration Note:
            This method provides a simplified interface for Sidekicks
            while handling all the LangChain integration complexity.
        """
        
        self.stats["manager_calls"] += 1
        
        logger.info(f"[{job_id}] LangChain Manager processing LLM call for Sidekick '{sidekick_name}'")
        
        try:
            # Prepare parameters for the RelayerLLM call
            call_params = {
                "job_id": job_id,
                "task_type": task_type,
                "sidekick_name": sidekick_name,
                "model_preference": kwargs.get("model_preference", self.default_model),
                "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
                "temperature": kwargs.get("temperature", self.default_temperature),
                "context_metadata": kwargs.get("context_metadata", {})
            }
            
            # Make the LLM call through our RelayerLLM
            llm_output = self.relayer_llm._call(prompt_text, **call_params)
            
            # Prepare structured response
            response = {
                "llm_output": llm_output,
                "model_used": call_params["model_preference"],
                "task_type": task_type,
                "sidekick_name": sidekick_name,
                "success": True
            }
            
            self.stats["successful_calls"] += 1
            logger.info(f"[{job_id}] LLM call successful")
            
            return response
            
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"[{job_id}] LLM call failed: {str(e)}")
            
            # Return error response
            return {
                "llm_output": "",
                "error": str(e),
                "success": False
            }
    
    def execute_llm_call(self, prompt_text: str, task_type: str, sidekick_name: str,
                        job_id: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an LLM call with full error handling and retry logic.
        
        This method provides a more robust interface for critical LLM calls
        that need retry logic and detailed error handling.
        
        Args:
            prompt_text: The prompt to send to the LLM
            task_type: Type of task being performed
            sidekick_name: Name of the calling Sidekick
            job_id: Job ID for logging and traceability
            **kwargs: Additional parameters including retry settings
            
        Returns:
            Dict[str, Any]: Detailed response with execution metadata
        """
        
        max_retries = kwargs.get("max_retries", 2)
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Make the LLM call
                result = self.call_llm(
                    prompt_text=prompt_text,
                    task_type=task_type,
                    sidekick_name=sidekick_name,
                    job_id=job_id,
                    **kwargs
                )
                
                if result["success"]:
                    # Add execution metadata
                    result["retry_count"] = retry_count
                    result["execution_time_ms"] = kwargs.get("execution_time_ms", 0)
                    return result
                else:
                    raise Exception(result.get("error", "Unknown error"))
                    
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"[{job_id}] All retries exhausted for LLM call")
                    return {
                        "llm_output": "",
                        "error": f"LLM call failed after {max_retries} retries: {str(e)}",
                        "success": False,
                        "retry_count": retry_count
                    }
                else:
                    logger.warning(f"[{job_id}] Retrying LLM call (attempt {retry_count + 1})")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the LangChain Manager.
        
        Returns:
            Dict[str, Any]: Status information including stats and health
        """
        
        return {
            "status": "active",
            "stats": self.stats.copy(),
            "llm_stats": self.relayer_llm.get_stats(),
            "config": {
                "default_model": self.default_model,
                "default_max_tokens": self.default_max_tokens,
                "default_temperature": self.default_temperature
            }
        }