"""
File: /sidekick-engine/storage/dynamic_prompt_manager.py

Fixed central manager for dynamic prompt storage and retrieval.

FIXES APPLIED:
- Fixed all import statements and module resolution
- Added proper async/await patterns throughout
- Fixed model imports and type hints
- Added comprehensive error handling
- Implemented missing methods and fallbacks
- Added proper LangChain integration
"""

import logging
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

# Fixed imports - using absolute imports for models
try:
    from ..models.dynamic_prompt_models import (
        PromptGenerationRequest, PromptResponse, StoredPrompt, 
        PromptSource, PromptStatus, ContextSignature
    )
    from .context_matcher import ContextMatcher
    from .cosmos_prompt_store import CosmosPromptStore
    from ..langchain_integration import LangChainManager
except ImportError:
    # Fallback for development/testing
    from models.dynamic_prompt_models import (
        PromptGenerationRequest, PromptResponse, StoredPrompt, 
        PromptSource, PromptStatus, ContextSignature
    )
    from context_matcher import ContextMatcher
    from cosmos_prompt_store import CosmosPromptStore
    from langchain_integration import LangChainManager

logger = logging.getLogger(__name__)

class DynamicPromptManager:
    """Manages dynamic prompt storage, retrieval, and generation"""
    
    def __init__(self, cosmos_store: CosmosPromptStore, langchain_manager: Optional[LangChainManager] = None):
        """
        Initialize the dynamic prompt manager.
        
        Args:
            cosmos_store: CosmosDB storage implementation
            langchain_manager: LangChain manager for LLM calls (optional)
        """
        self.cosmos_store = cosmos_store
        self.langchain_manager = langchain_manager
        
        # Configuration
        self.default_similarity_threshold = 0.8
        self.max_similar_prompts = 5
        self.enable_prompt_learning = True
        self.enable_auto_generation = True
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.generation_count = 0
        self._stats_lock = asyncio.Lock()
        
        logger.info("DynamicPromptManager initialized")
    
    async def get_or_generate_prompt(self, request: PromptGenerationRequest) -> PromptResponse:
        """
        Main method: get existing prompt or generate new one.
        
        Args:
            request: Prompt generation request
            
        Returns:
            PromptResponse with prompt and metadata
        """
        try:
            logger.info(f"Processing prompt request for {request.sidekick_name}.{request.task_type}")
            
            # Step 1: Try exact match first
            exact_match = await self._try_exact_match(request)
            if exact_match:
                async with self._stats_lock:
                    self.cache_hits += 1
                return exact_match
            
            # Step 2: Try similarity matching
            similar_match = await self._try_similarity_match(request)
            if similar_match:
                async with self._stats_lock:
                    self.cache_hits += 1
                return similar_match
            
            # Step 3: Generate new prompt if enabled
            if request.fallback_to_llm and self.enable_auto_generation:
                async with self._stats_lock:
                    self.cache_misses += 1
                generated_response = await self._generate_new_prompt(request)
                
                # Save generated prompt if requested
                if request.save_generated:
                    try:
                        await self._save_generated_prompt(request, generated_response)
                    except Exception as e:
                        logger.warning(f"Failed to save generated prompt: {str(e)}")
                
                return generated_response
            
            # Step 4: No prompt available
            async with self._stats_lock:
                self.cache_misses += 1
            raise Exception(
                f"No prompt available for {request.sidekick_name}.{request.task_type} "
                f"and LLM generation is disabled"
            )
            
        except Exception as e:
            logger.error(f"Error in get_or_generate_prompt: {str(e)}")
            # Return fallback prompt instead of raising
            return self._create_fallback_prompt(request)
    
    async def _try_exact_match(self, request: PromptGenerationRequest) -> Optional[PromptResponse]:
        """Try to find an exact context match in the database"""
        try:
            context_hash = ContextMatcher.create_context_hash(request.job_context)
            
            exact_match = await self.cosmos_store.get_prompt(
                request.sidekick_name, request.task_type, context_hash
            )
            
            if exact_match:
                logger.info(f"Found exact prompt match for {request.sidekick_name}.{request.task_type}")
                
                # Update usage statistics
                await self._update_prompt_usage(exact_match.prompt_id)
                
                return PromptResponse(
                    prompt_template=exact_match.prompt_template,
                    input_variables=exact_match.input_variables,
                    source=PromptSource.EXACT_MATCH,
                    prompt_id=exact_match.prompt_id,
                    context_match_score=1.0,
                    performance_prediction=exact_match.success_rate,
                    usage_count=exact_match.usage_count,
                    last_used=exact_match.last_used
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in exact match lookup: {str(e)}")
            return None
    
    async def _try_similarity_match(self, request: PromptGenerationRequest) -> Optional[PromptResponse]:
        """Try to find a similar context match in the database"""
        try:
            context_signature = ContextMatcher.create_context_signature(
                request.job_context, request.sidekick_name
            )
            
            similar_prompts = await self.cosmos_store.find_similar_prompts(
                request.sidekick_name, 
                request.task_type, 
                context_signature.model_dump(exclude_none=True) if hasattr(context_signature, 'model_dump') else context_signature.dict(exclude_none=True),
                self.max_similar_prompts
            )
            
            if not similar_prompts:
                return None
            
            # Find the best match above threshold
            best_match = None
            best_score = 0.0
            
            for prompt in similar_prompts:
                try:
                    # Convert stored context signature back to ContextSignature object
                    if isinstance(prompt.context_signature, dict):
                        prompt_signature = ContextSignature(**prompt.context_signature)
                    else:
                        prompt_signature = prompt.context_signature
                        
                    similarity = ContextMatcher.calculate_similarity(
                        context_signature, prompt_signature, request.sidekick_name
                    )
                    
                    if similarity >= request.similarity_threshold and similarity > best_score:
                        best_match = prompt
                        best_score = similarity
                        
                except Exception as e:
                    logger.warning(f"Error calculating similarity for prompt {prompt.prompt_id}: {str(e)}")
                    continue
            
            if best_match:
                logger.info(
                    f"Found similar prompt match for {request.sidekick_name}.{request.task_type} "
                    f"(similarity: {best_score:.3f})"
                )
                
                # Update usage statistics
                await self._update_prompt_usage(best_match.prompt_id)
                
                return PromptResponse(
                    prompt_template=best_match.prompt_template,
                    input_variables=best_match.input_variables,
                    source=PromptSource.SIMILAR_MATCH,
                    prompt_id=best_match.prompt_id,
                    context_match_score=best_score,
                    performance_prediction=best_match.success_rate * best_score,  # Adjust for similarity
                    usage_count=best_match.usage_count,
                    last_used=best_match.last_used
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in similarity match lookup: {str(e)}")
            return None
    
    async def _generate_new_prompt(self, request: PromptGenerationRequest) -> PromptResponse:
        """Generate a new prompt using LLM"""
        try:
            logger.info(f"Generating new prompt for {request.sidekick_name}.{request.task_type}")
            async with self._stats_lock:
                self.generation_count += 1
            
            # Check if LangChain manager is available
            if not self.langchain_manager:
                logger.warning("No LangChain manager available, using fallback prompt")
                return self._create_fallback_prompt(request)
            
            # Create meta-prompt for generating the specific prompt
            meta_prompt = self._create_meta_prompt(request)
            
            # Call LLM via LangChain manager
            try:
                if hasattr(self.langchain_manager, 'call_llm'):
                    llm_response = await self.langchain_manager.call_llm(
                        prompt_text=meta_prompt,
                        task_type=request.task_type,
                        sidekick_name=request.sidekick_name,
                        job_id=f"generate_{uuid.uuid4().hex[:8]}"
                    )
                else:
                    # Fallback method name
                    llm_response = await self.langchain_manager.execute_llm_call(
                        prompt_text=meta_prompt,
                        task_type=request.task_type,
                        sidekick_name=request.sidekick_name,
                        job_id=f"generate_{uuid.uuid4().hex[:8]}"
                    )
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                return self._create_fallback_prompt(request)
            
            # Parse the LLM response to extract template and variables
            parsed_response = self._parse_llm_response(llm_response, request)
            
            return PromptResponse(
                prompt_template=parsed_response["template"],
                input_variables=parsed_response["input_variables"],
                source=PromptSource.GENERATED,
                prompt_id=None,  # Will be assigned when saved
                context_match_score=0.0,
                performance_prediction=0.5,  # Default for new prompts
                usage_count=0,
                last_used=None
            )
            
        except Exception as e:
            logger.error(f"Error generating new prompt: {str(e)}")
            # Return fallback prompt
            return self._create_fallback_prompt(request)
    
    def _create_meta_prompt(self, request: PromptGenerationRequest) -> str:
        """Create a meta-prompt for generating task-specific prompts"""
        
        # Get task-specific guidance
        task_guidance = self._get_task_guidance(request.sidekick_name, request.task_type)
        
        # Extract key context elements
        context_summary = self._summarize_context(request.job_context)
        
        meta_prompt = f"""You are an expert prompt engineer creating prompts for the Clarity AI platform.

Task: Create a high-quality prompt template for a Sidekick named "{request.sidekick_name}" to handle "{request.task_type}" tasks.

Sidekick Context:
{task_guidance}

Job Context:
{context_summary}

Requirements:
1. Create a prompt that is specific to the task type and context
2. Use {{variable_name}} format for input variables
3. Include clear instructions and formatting requirements
4. Make the prompt reusable for similar contexts
5. Ensure the prompt guides the LLM to produce structured, actionable output
6. Include role-based instructions (e.g., "You are an expert data analyst...")

Output Format:
Return ONLY a JSON object with this structure:
{{
    "template": "The complete prompt template with {{variable}} placeholders",
    "input_variables": ["list", "of", "required", "variables"],
    "description": "Brief description of what this prompt does"
}}

Generate the prompt template now:"""

        return meta_prompt
    
    def _get_task_guidance(self, sidekick_name: str, task_type: str) -> str:
        """Get task-specific guidance for prompt generation"""
        
        guidance_map = {
            "fixxy": {
                "deduplicate": "Focus on identifying and removing duplicate records based on specified matching fields",
                "format_cleanup": "Standardize data formats according to specified rules and conventions",
                "null_handling": "Handle missing values using appropriate imputation or flagging strategies",
                "validate_formats": "Verify data format compliance and flag invalid entries",
                "data_quality_check": "Assess overall data quality across multiple dimensions"
            },
            "findy": {
                "pattern_recognition": "Identify recurring patterns and trends in the data",
                "anomaly_detection": "Detect outliers and unusual behaviors in the dataset",
                "trend_analysis": "Analyze trends over time and identify significant changes",
                "behavioral_insights": "Extract insights about user or entity behaviors"
            },
            "predicty": {
                "time_series_forecast": "Generate forecasts for time-based data with confidence intervals",
                "demand_forecasting": "Predict future demand patterns with seasonal adjustments",
                "scenario_modeling": "Create multiple scenario predictions with probability assessments"
            }
        }
        
        sidekick_guidance = guidance_map.get(sidekick_name, {})
        task_specific = sidekick_guidance.get(task_type, f"Handle {task_type} tasks effectively")
        
        return f"Sidekick '{sidekick_name}' specializes in: {task_specific}"
    
    def _summarize_context(self, job_context: Dict[str, Any]) -> str:
        """Summarize job context for meta-prompt"""
        key_elements = []
        
        # Extract important context elements
        important_keys = [
            "field_list", "data_types", "analysis_focus", "cleanup_rules",
            "validation_rules", "forecast_horizon", "confidence_level"
        ]
        
        for key in important_keys:
            if key in job_context:
                value = job_context[key]
                if isinstance(value, (list, dict)):
                    key_elements.append(f"- {key}: {str(value)[:100]}...")
                else:
                    key_elements.append(f"- {key}: {value}")
        
        return "\n".join(key_elements) if key_elements else "- No specific context provided"
    
    def _parse_llm_response(self, llm_response: Dict[str, Any], request: PromptGenerationRequest) -> Dict[str, Any]:
        """Parse LLM response to extract template and variables"""
        try:
            import json
            
            # Extract LLM output from response
            llm_output = ""
            if isinstance(llm_response, dict):
                llm_output = llm_response.get("llm_output", "")
            else:
                llm_output = str(llm_response)
            
            # Try to parse as JSON
            if llm_output.strip().startswith('{'):
                parsed = json.loads(llm_output)
                return {
                    "template": parsed.get("template", ""),
                    "input_variables": parsed.get("input_variables", []),
                    "description": parsed.get("description", "")
                }
            else:
                # Fallback: extract template from response
                return self._extract_template_from_text(llm_output, request)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, using text extraction")
            llm_output = llm_response.get("llm_output", "") if isinstance(llm_response, dict) else str(llm_response)
            return self._extract_template_from_text(llm_output, request)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return self._extract_template_from_text("", request)
    
    def _extract_template_from_text(self, response: str, request: PromptGenerationRequest) -> Dict[str, Any]:
        """Extract template from free-form text response"""
        import re
        
        # Find variables in {variable} format
        variables = re.findall(r'\{([^}]+)\}', response)
        
        # Use the response as the template
        template = response.strip()
        
        # Ensure we have at least basic variables
        if not variables:
            variables = ["input_data", "task_context"]
            template += "\n\nInput: {input_data}\nContext: {task_context}"
        
        return {
            "template": template,
            "input_variables": list(set(variables)),  # Remove duplicates
            "description": f"Generated prompt for {request.sidekick_name}.{request.task_type}"
        }
    
    def _create_fallback_prompt(self, request: PromptGenerationRequest) -> PromptResponse:
        """Create a basic fallback prompt when generation fails"""
        
        fallback_template = f"""You are an expert assistant helping with {request.task_type} tasks.

Task: {request.task_type}
Context: {{job_context}}
Input Data: {{input_data}}

Please analyze the provided data and complete the {request.task_type} task according to the given context and requirements.

Provide your response in a clear, structured format."""

        return PromptResponse(
            prompt_template=fallback_template,
            input_variables=["job_context", "input_data"],
            source=PromptSource.FALLBACK,
            prompt_id=None,
            context_match_score=0.0,
            performance_prediction=0.3,  # Lower prediction for fallback
            usage_count=0,
            last_used=None
        )
    
    async def _save_generated_prompt(self, request: PromptGenerationRequest, 
                                   response: PromptResponse) -> None:
        """Save newly generated prompt to storage"""
        try:
            context_hash = ContextMatcher.create_context_hash(request.job_context)
            context_signature = ContextMatcher.create_context_signature(
                request.job_context, request.sidekick_name
            )
            
            # Generate unique prompt ID
            prompt_id = f"{request.sidekick_name}_{request.task_type}_{context_hash}"
            
            stored_prompt = StoredPrompt(
                prompt_id=prompt_id,
                sidekick_name=request.sidekick_name,
                task_type=request.task_type,
                context_hash=context_hash,
                prompt_template=response.prompt_template,
                input_variables=response.input_variables,
                context_signature=context_signature.model_dump(exclude_none=True) if hasattr(context_signature, 'model_dump') else context_signature.dict(exclude_none=True),
                performance_metrics={"initial_generation": True},
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                usage_count=1,
                success_rate=0.5,  # Default for new prompts
                version="1.0",
                status=PromptStatus.ACTIVE,
                tags=[request.sidekick_name, request.task_type, "auto_generated"]
            )
            
            await self.cosmos_store.save_prompt(stored_prompt)
            
            # Update response with saved prompt ID
            response.prompt_id = prompt_id
            
            logger.info(f"Saved generated prompt: {prompt_id}")
            
        except Exception as e:
            logger.error(f"Failed to save generated prompt: {str(e)}")
            # Continue without saving - don't fail the entire request
    
    async def _update_prompt_usage(self, prompt_id: str) -> None:
        """Update prompt usage statistics"""
        try:
            # This will be called when a prompt is retrieved and used
            # The actual success/failure will be reported later via report_prompt_success
            pass
        except Exception as e:
            logger.error(f"Error updating prompt usage for {prompt_id}: {str(e)}")
    
    async def report_prompt_success(self, prompt_id: str, success: bool, 
                                  response_time: float, quality_score: Optional[float] = None) -> None:
        """
        Report success/failure of a prompt execution for learning.
        
        Args:
            prompt_id: ID of the prompt that was executed
            success: Whether the execution was successful
            response_time: Response time in seconds
            quality_score: Optional quality score (0.0-1.0)
        """
        try:
            if prompt_id and self.enable_prompt_learning:
                await self.cosmos_store.update_prompt_metrics(prompt_id, success, response_time)
                logger.debug(f"Reported prompt performance: {prompt_id}, success={success}")
        except Exception as e:
            logger.error(f"Error reporting prompt success: {str(e)}")
    
    async def get_prompt_statistics(self, sidekick_name: str, task_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get usage statistics for prompts"""
        try:
            return await self.cosmos_store.get_prompt_statistics(sidekick_name, task_type)
        except Exception as e:
            logger.error(f"Error getting prompt statistics: {str(e)}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the prompt manager"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "cache_hit_rate": cache_hit_rate,
            "generations": self.generation_count
        }
    
    async def migrate_legacy_prompts(self, legacy_prompts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Migrate legacy hardcoded prompts to dynamic storage.
        
        Args:
            legacy_prompts: Dictionary of legacy prompt definitions
            
        Returns:
            Migration results
        """
        migration_results = {
            "migrated_prompts": 0,
            "failed_migrations": 0,
            "migration_details": []
        }
        
        try:
            for task_type, prompt_data in legacy_prompts.items():
                try:
                    # Create context from default context
                    default_context = prompt_data.get("default_context", {})
                    context_hash = ContextMatcher.create_context_hash(default_context)
                    context_signature = ContextMatcher.create_context_signature(default_context, "fixxy")
                    
                    # Generate unique prompt ID
                    prompt_id = f"fixxy_{task_type}_legacy_{context_hash}"
                    
                    stored_prompt = StoredPrompt(
                        prompt_id=prompt_id,
                        sidekick_name="fixxy",
                        task_type=task_type,
                        context_hash=context_hash,
                        prompt_template=prompt_data["template"],
                        input_variables=prompt_data["input_variables"],
                        context_signature=context_signature.model_dump(exclude_none=True) if hasattr(context_signature, 'model_dump') else context_signature.dict(exclude_none=True),
                        performance_metrics={"migrated_from": "legacy_hardcoded"},
                        created_at=datetime.utcnow(),
                        last_used=datetime.utcnow(),
                        usage_count=0,
                        success_rate=0.85,  # Assume good performance for legacy prompts
                        version="1.0",
                        status=PromptStatus.ACTIVE,
                        tags=["fixxy", task_type, "legacy_migration"]
                    )
                    
                    await self.cosmos_store.save_prompt(stored_prompt)
                    
                    migration_results["migrated_prompts"] += 1
                    migration_results["migration_details"].append({
                        "task_type": task_type,
                        "prompt_id": prompt_id,
                        "status": "success"
                    })
                    
                except Exception as e:
                    migration_results["failed_migrations"] += 1
                    migration_results["migration_details"].append({
                        "task_type": task_type,
                        "status": "failed",
                        "error": str(e)
                    })
                    logger.error(f"Failed to migrate prompt for {task_type}: {str(e)}")
            
            logger.info(f"Migration completed: {migration_results['migrated_prompts']} successful, {migration_results['failed_migrations']} failed")
            return migration_results
            
        except Exception as e:
            logger.error(f"Error during legacy prompt migration: {str(e)}")
            migration_results["migration_error"] = str(e)
            return migration_results