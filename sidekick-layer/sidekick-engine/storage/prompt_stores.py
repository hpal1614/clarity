"""
storage/prompt_stores.py - Missing Storage Classes

This file implements ALL missing storage classes referenced in the codebase,
ensuring zero compilation errors for Azure deployment.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import hashlib
from abc import ABC, abstractmethod

# Try importing Azure Cosmos (will be available in Azure runtime)
try:
    from azure.cosmos.aio import CosmosClient
    from azure.cosmos import PartitionKey
    from azure.cosmos.exceptions import CosmosResourceNotFoundError, CosmosResourceExistsError
    COSMOS_AVAILABLE = True
except ImportError:
    # Fallback for development environments
    COSMOS_AVAILABLE = False
    logging.warning("Azure Cosmos SDK not available - using in-memory storage")

from ..models import (
    StoredPrompt, PromptStatus, ContextSignature,
    create_context_signature_from_dict, generate_context_hash
)

logger = logging.getLogger(__name__)

# ========================================
# BASE STORAGE INTERFACE
# ========================================

class BasePromptStore(ABC):
    """Base interface for prompt storage implementations"""
    
    @abstractmethod
    async def save_prompt(self, prompt: StoredPrompt) -> str:
        """Save a prompt and return its ID"""
        pass
    
    @abstractmethod
    async def get_prompt(self, sidekick_name: str, task_type: str, context_hash: str) -> Optional[StoredPrompt]:
        """Get a prompt by exact context match"""
        pass
    
    @abstractmethod
    async def find_similar_prompts(self, sidekick_name: str, task_type: str, 
                                 context_signature: Dict[str, Any], limit: int = 5) -> List[StoredPrompt]:
        """Find prompts with similar contexts"""
        pass
    
    @abstractmethod
    async def update_prompt_metrics(self, prompt_id: str, success: bool, response_time: float) -> bool:
        """Update prompt performance metrics"""
        pass
    
    @abstractmethod
    async def get_prompt_statistics(self, sidekick_name: str, task_type: str = None) -> Optional[Dict[str, Any]]:
        """Get usage statistics for prompts"""
        pass

# ========================================
# IN-MEMORY STORAGE (MISSING CLASS)
# ========================================

class FixxyPromptStore(BasePromptStore):
    """
    In-memory prompt storage for FixxySidekick (MISSING - Referenced in dynamic_fixxy.py)
    
    This class provides fast in-memory storage for development and testing.
    In production, use CosmosFixxyPromptStore for persistent storage.
    """
    
    def __init__(self):
        """Initialize in-memory storage"""
        self.prompts: Dict[str, StoredPrompt] = {}
        self.context_index: Dict[str, List[str]] = {}  # context_hash -> prompt_ids
        self.task_index: Dict[str, Dict[str, List[str]]] = {}  # sidekick -> task_type -> prompt_ids
        self.stats = {
            "total_prompts": 0,
            "total_retrievals": 0,
            "cache_hits": 0,
            "total_saves": 0
        }
        logger.info("Initialized FixxyPromptStore (in-memory)")
    
    async def save_prompt(self, prompt: StoredPrompt) -> str:
        """Save prompt to in-memory storage"""
        try:
            prompt_id = prompt.prompt_id
            
            # Store the prompt
            self.prompts[prompt_id] = prompt
            
            # Update context index
            if prompt.context_hash not in self.context_index:
                self.context_index[prompt.context_hash] = []
            if prompt_id not in self.context_index[prompt.context_hash]:
                self.context_index[prompt.context_hash].append(prompt_id)
            
            # Update task index
            if prompt.sidekick_name not in self.task_index:
                self.task_index[prompt.sidekick_name] = {}
            if prompt.task_type not in self.task_index[prompt.sidekick_name]:
                self.task_index[prompt.sidekick_name][prompt.task_type] = []
            if prompt_id not in self.task_index[prompt.sidekick_name][prompt.task_type]:
                self.task_index[prompt.sidekick_name][prompt.task_type].append(prompt_id)
            
            self.stats["total_saves"] += 1
            self.stats["total_prompts"] = len(self.prompts)
            
            logger.debug(f"Saved prompt {prompt_id} to in-memory store")
            return prompt_id
            
        except Exception as e:
            logger.error(f"Error saving prompt to in-memory store: {str(e)}")
            raise
    
    async def get_prompt(self, sidekick_name: str, task_type: str, context_hash: str) -> Optional[StoredPrompt]:
        """Get prompt by exact context match"""
        try:
            self.stats["total_retrievals"] += 1
            
            # Look up by context hash
            if context_hash in self.context_index:
                for prompt_id in self.context_index[context_hash]:
                    prompt = self.prompts.get(prompt_id)
                    if (prompt and 
                        prompt.sidekick_name == sidekick_name and 
                        prompt.task_type == task_type and
                        prompt.status == PromptStatus.ACTIVE):
                        
                        # Update last used
                        prompt.last_used = datetime.utcnow()
                        prompt.usage_count += 1
                        
                        self.stats["cache_hits"] += 1
                        logger.debug(f"Found exact match for {sidekick_name}.{task_type}")
                        return prompt
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving prompt: {str(e)}")
            return None
    
    async def find_similar_prompts(self, sidekick_name: str, task_type: str, 
                                 context_signature: Dict[str, Any], limit: int = 5) -> List[StoredPrompt]:
        """Find prompts with similar contexts"""
        try:
            similar_prompts = []
            
            # Get all prompts for this sidekick and task type
            if (sidekick_name in self.task_index and 
                task_type in self.task_index[sidekick_name]):
                
                prompt_ids = self.task_index[sidekick_name][task_type]
                
                for prompt_id in prompt_ids:
                    prompt = self.prompts.get(prompt_id)
                    if prompt and prompt.status == PromptStatus.ACTIVE:
                        # Simple similarity calculation based on matching fields
                        similarity = self._calculate_similarity(context_signature, prompt.context_signature)
                        if similarity > 0.5:  # Basic threshold
                            similar_prompts.append((similarity, prompt))
                
                # Sort by similarity and return top results
                similar_prompts.sort(key=lambda x: x[0], reverse=True)
                result = [prompt for _, prompt in similar_prompts[:limit]]
                
                logger.debug(f"Found {len(result)} similar prompts for {sidekick_name}.{task_type}")
                return result
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding similar prompts: {str(e)}")
            return []
    
    def _calculate_similarity(self, sig1: Dict[str, Any], sig2: Dict[str, Any]) -> float:
        """Simple similarity calculation"""
        try:
            if not sig1 or not sig2:
                return 0.0
            
            matches = 0
            total = 0
            
            all_keys = set(sig1.keys()) | set(sig2.keys())
            for key in all_keys:
                total += 1
                if key in sig1 and key in sig2:
                    val1 = str(sig1[key]).lower() if sig1[key] else ""
                    val2 = str(sig2[key]).lower() if sig2[key] else ""
                    if val1 == val2:
                        matches += 1
                    elif val1 and val2 and (val1 in val2 or val2 in val1):
                        matches += 0.5
            
            return matches / total if total > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def update_prompt_metrics(self, prompt_id: str, success: bool, response_time: float) -> bool:
        """Update prompt performance metrics"""
        try:
            if prompt_id in self.prompts:
                prompt = self.prompts[prompt_id]
                
                # Update usage count
                prompt.usage_count += 1
                
                # Update success rate
                if success:
                    old_successes = int(prompt.success_rate * (prompt.usage_count - 1))
                    prompt.success_rate = (old_successes + 1) / prompt.usage_count
                else:
                    old_successes = int(prompt.success_rate * (prompt.usage_count - 1))
                    prompt.success_rate = old_successes / prompt.usage_count
                
                # Update average response time
                old_avg = prompt.avg_response_time
                prompt.avg_response_time = ((old_avg * (prompt.usage_count - 1)) + response_time) / prompt.usage_count
                
                # Update timestamp
                prompt.last_used = datetime.utcnow()
                
                logger.debug(f"Updated metrics for prompt {prompt_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating prompt metrics: {str(e)}")
            return False
    
    async def get_prompt_statistics(self, sidekick_name: str, task_type: str = None) -> Optional[Dict[str, Any]]:
        """Get usage statistics for prompts"""
        try:
            stats = {
                "sidekick_name": sidekick_name,
                "task_type": task_type,
                "total_prompts": 0,
                "active_prompts": 0,
                "total_usage_count": 0,
                "avg_success_rate": 0.0,
                "avg_response_time": 0.0
            }
            
            # Filter prompts
            relevant_prompts = []
            for prompt in self.prompts.values():
                if prompt.sidekick_name == sidekick_name:
                    if task_type is None or prompt.task_type == task_type:
                        relevant_prompts.append(prompt)
            
            if not relevant_prompts:
                return None
            
            # Calculate statistics
            stats["total_prompts"] = len(relevant_prompts)
            stats["active_prompts"] = len([p for p in relevant_prompts if p.status == PromptStatus.ACTIVE])
            stats["total_usage_count"] = sum(p.usage_count for p in relevant_prompts)
            
            if relevant_prompts:
                stats["avg_success_rate"] = sum(p.success_rate for p in relevant_prompts) / len(relevant_prompts)
                stats["avg_response_time"] = sum(p.avg_response_time for p in relevant_prompts) / len(relevant_prompts)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting prompt statistics: {str(e)}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get storage performance statistics"""
        cache_hit_rate = self.stats["cache_hits"] / self.stats["total_retrievals"] if self.stats["total_retrievals"] > 0 else 0
        
        return {
            "storage_type": "in_memory",
            "total_prompts": self.stats["total_prompts"],
            "total_saves": self.stats["total_saves"],
            "total_retrievals": self.stats["total_retrievals"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": cache_hit_rate
        }

# ========================================
# COSMOS DB STORAGE (MISSING CLASS)
# ========================================

class CosmosFixxyPromptStore(BasePromptStore):
    """
    Azure CosmosDB storage for FixxySidekick prompts (MISSING - Referenced in main_dynamic.py)
    
    This class provides production-grade persistent storage using Azure CosmosDB.
    """
    
    def __init__(self, connection_string: str):
        """Initialize Cosmos DB storage"""
        self.connection_string = connection_string
        self.database_name = "clarity_prompts"
        self.container_name = "stored_prompts"
        self.client = None
        self.database = None
        self.container = None
        self.initialized = False
        
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "last_error": None
        }
        
        logger.info("Initialized CosmosFixxyPromptStore")
    
    async def initialize(self):
        """Initialize Cosmos DB connection"""
        if self.initialized:
            return
        
        try:
            if not COSMOS_AVAILABLE:
                raise Exception("Azure Cosmos SDK not available")
            
            # Create client
            self.client = CosmosClient.from_connection_string(self.connection_string)
            
            # Get database
            self.database = self.client.get_database_client(self.database_name)
            
            # Get container
            self.container = self.database.get_container_client(self.container_name)
            
            self.initialized = True
            logger.info("CosmosDB connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CosmosDB: {str(e)}")
            self.stats["last_error"] = str(e)
            raise
    
    async def save_prompt(self, prompt: StoredPrompt) -> str:
        """Save prompt to CosmosDB"""
        try:
            await self.initialize()
            self.stats["total_operations"] += 1
            
            # Convert to CosmosDB format
            item = {
                "id": prompt.prompt_id,
                "prompt_id": prompt.prompt_id,
                "sidekick_name": prompt.sidekick_name,
                "task_type": prompt.task_type,
                "context_hash": prompt.context_hash,
                "prompt_template": prompt.prompt_template,
                "input_variables": prompt.input_variables,
                "context_signature": prompt.context_signature,
                "performance_metrics": prompt.performance_metrics,
                "created_at": prompt.created_at.isoformat(),
                "last_used": prompt.last_used.isoformat(),
                "usage_count": prompt.usage_count,
                "success_rate": prompt.success_rate,
                "avg_response_time": prompt.avg_response_time,
                "version": prompt.version,
                "status": prompt.status.value,
                "tags": prompt.tags
            }
            
            # Save to CosmosDB
            await self.container.upsert_item(body=item)
            
            self.stats["successful_operations"] += 1
            logger.debug(f"Saved prompt {prompt.prompt_id} to CosmosDB")
            return prompt.prompt_id
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            logger.error(f"Error saving prompt to CosmosDB: {str(e)}")
            raise
    
    async def get_prompt(self, sidekick_name: str, task_type: str, context_hash: str) -> Optional[StoredPrompt]:
        """Get prompt by exact context match from CosmosDB"""
        try:
            await self.initialize()
            self.stats["total_operations"] += 1
            
            # Query for prompt with exact context match
            query = """
            SELECT * FROM c 
            WHERE c.sidekick_name = @sidekick_name 
            AND c.task_type = @task_type 
            AND c.context_hash = @context_hash 
            AND c.status = @status
            """
            
            parameters = [
                {"name": "@sidekick_name", "value": sidekick_name},
                {"name": "@task_type", "value": task_type},
                {"name": "@context_hash", "value": context_hash},
                {"name": "@status", "value": PromptStatus.ACTIVE.value}
            ]
            
            items = []
            async for item in self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=sidekick_name
            ):
                items.append(item)
            
            if items:
                # Convert back to StoredPrompt
                item = items[0]  # Take first match
                prompt = self._item_to_stored_prompt(item)
                
                # Update usage metrics
                await self.update_prompt_metrics(prompt.prompt_id, True, 0.0)
                
                self.stats["successful_operations"] += 1
                logger.debug(f"Retrieved prompt {prompt.prompt_id} from CosmosDB")
                return prompt
            
            self.stats["successful_operations"] += 1
            return None
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            logger.error(f"Error retrieving prompt from CosmosDB: {str(e)}")
            return None
    
    async def find_similar_prompts(self, sidekick_name: str, task_type: str, 
                                 context_signature: Dict[str, Any], limit: int = 5) -> List[StoredPrompt]:
        """Find similar prompts from CosmosDB"""
        try:
            await self.initialize()
            self.stats["total_operations"] += 1
            
            # Query for prompts of same sidekick and task type
            query = """
            SELECT * FROM c 
            WHERE c.sidekick_name = @sidekick_name 
            AND c.task_type = @task_type 
            AND c.status = @status
            ORDER BY c.usage_count DESC
            """
            
            parameters = [
                {"name": "@sidekick_name", "value": sidekick_name},
                {"name": "@task_type", "value": task_type},
                {"name": "@status", "value": PromptStatus.ACTIVE.value}
            ]
            
            similar_prompts = []
            async for item in self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=sidekick_name
            ):
                prompt = self._item_to_stored_prompt(item)
                
                # Calculate similarity (simple implementation)
                similarity = self._calculate_similarity(context_signature, prompt.context_signature)
                if similarity > 0.5:  # Basic threshold
                    similar_prompts.append((similarity, prompt))
                
                if len(similar_prompts) >= limit * 2:  # Get more to sort
                    break
            
            # Sort by similarity and return top results
            similar_prompts.sort(key=lambda x: x[0], reverse=True)
            result = [prompt for _, prompt in similar_prompts[:limit]]
            
            self.stats["successful_operations"] += 1
            logger.debug(f"Found {len(result)} similar prompts in CosmosDB")
            return result
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            logger.error(f"Error finding similar prompts in CosmosDB: {str(e)}")
            return []
    
    def _calculate_similarity(self, sig1: Dict[str, Any], sig2: Dict[str, Any]) -> float:
        """Calculate similarity between context signatures"""
        try:
            if not sig1 or not sig2:
                return 0.0
            
            matches = 0
            total = 0
            
            all_keys = set(sig1.keys()) | set(sig2.keys())
            for key in all_keys:
                total += 1
                if key in sig1 and key in sig2:
                    val1 = str(sig1[key]).lower() if sig1[key] else ""
                    val2 = str(sig2[key]).lower() if sig2[key] else ""
                    if val1 == val2:
                        matches += 1
                    elif val1 and val2 and (val1 in val2 or val2 in val1):
                        matches += 0.5
            
            return matches / total if total > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def update_prompt_metrics(self, prompt_id: str, success: bool, response_time: float) -> bool:
        """Update prompt metrics in CosmosDB"""
        try:
            await self.initialize()
            self.stats["total_operations"] += 1
            
            # Get current prompt
            try:
                item = await self.container.read_item(
                    item=prompt_id,
                    partition_key=prompt_id.split('_')[0]  # Extract sidekick name
                )
            except CosmosResourceNotFoundError:
                logger.warning(f"Prompt {prompt_id} not found for metrics update")
                return False
            
            # Update metrics
            usage_count = item.get("usage_count", 0) + 1
            current_success_rate = item.get("success_rate", 0.5)
            current_avg_time = item.get("avg_response_time", 3.0)
            
            # Calculate new success rate
            if success:
                old_successes = int(current_success_rate * (usage_count - 1))
                new_success_rate = (old_successes + 1) / usage_count
            else:
                old_successes = int(current_success_rate * (usage_count - 1))
                new_success_rate = old_successes / usage_count
            
            # Calculate new average response time
            new_avg_time = ((current_avg_time * (usage_count - 1)) + response_time) / usage_count
            
            # Update item
            item["usage_count"] = usage_count
            item["success_rate"] = new_success_rate
            item["avg_response_time"] = new_avg_time
            item["last_used"] = datetime.utcnow().isoformat()
            
            await self.container.replace_item(item=item, body=item)
            
            self.stats["successful_operations"] += 1
            logger.debug(f"Updated metrics for prompt {prompt_id}")
            return True
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            logger.error(f"Error updating prompt metrics: {str(e)}")
            return False
    
    async def get_prompt_statistics(self, sidekick_name: str, task_type: str = None) -> Optional[Dict[str, Any]]:
        """Get prompt statistics from CosmosDB"""
        try:
            await self.initialize()
            self.stats["total_operations"] += 1
            
            # Build query
            if task_type:
                query = """
                SELECT 
                    COUNT(1) as total_prompts,
                    SUM(c.usage_count) as total_usage,
                    AVG(c.success_rate) as avg_success_rate,
                    AVG(c.avg_response_time) as avg_response_time
                FROM c 
                WHERE c.sidekick_name = @sidekick_name 
                AND c.task_type = @task_type
                """
                parameters = [
                    {"name": "@sidekick_name", "value": sidekick_name},
                    {"name": "@task_type", "value": task_type}
                ]
            else:
                query = """
                SELECT 
                    COUNT(1) as total_prompts,
                    SUM(c.usage_count) as total_usage,
                    AVG(c.success_rate) as avg_success_rate,
                    AVG(c.avg_response_time) as avg_response_time
                FROM c 
                WHERE c.sidekick_name = @sidekick_name
                """
                parameters = [
                    {"name": "@sidekick_name", "value": sidekick_name}
                ]
            
            results = []
            async for result in self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=sidekick_name
            ):
                results.append(result)
            
            if results:
                result = results[0]
                stats = {
                    "sidekick_name": sidekick_name,
                    "task_type": task_type,
                    "total_prompts": result.get("total_prompts", 0),
                    "active_prompts": result.get("total_prompts", 0),  # Simplified
                    "total_usage_count": result.get("total_usage", 0),
                    "avg_success_rate": result.get("avg_success_rate", 0.0),
                    "avg_response_time": result.get("avg_response_time", 0.0)
                }
                
                self.stats["successful_operations"] += 1
                return stats
            
            return None
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            logger.error(f"Error getting prompt statistics: {str(e)}")
            return None
    
    def _item_to_stored_prompt(self, item: Dict[str, Any]) -> StoredPrompt:
        """Convert CosmosDB item to StoredPrompt"""
        return StoredPrompt(
            prompt_id=item["prompt_id"],
            sidekick_name=item["sidekick_name"],
            task_type=item["task_type"],
            context_hash=item["context_hash"],
            prompt_template=item["prompt_template"],
            input_variables=item.get("input_variables", []),
            context_signature=item.get("context_signature", {}),
            performance_metrics=item.get("performance_metrics", {}),
            created_at=datetime.fromisoformat(item["created_at"].replace('Z', '+00:00')) if isinstance(item["created_at"], str) else item["created_at"],
            last_used=datetime.fromisoformat(item["last_used"].replace('Z', '+00:00')) if isinstance(item["last_used"], str) else item["last_used"],
            usage_count=item.get("usage_count", 0),
            success_rate=item.get("success_rate", 0.5),
            avg_response_time=item.get("avg_response_time", 3.0),
            version=item.get("version", "1.0"),
            status=PromptStatus(item.get("status", PromptStatus.ACTIVE.value)),
            tags=item.get("tags", [])
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get storage performance statistics"""
        success_rate = 0.0
        if self.stats["total_operations"] > 0:
            success_rate = self.stats["successful_operations"] / self.stats["total_operations"]
        
        return {
            "storage_type": "cosmos_db",
            "total_operations": self.stats["total_operations"],
            "successful_operations": self.stats["successful_operations"],
            "failed_operations": self.stats["failed_operations"],
            "success_rate": success_rate,
            "last_error": self.stats["last_error"],
            "initialized": self.initialized
        }

# ========================================
# MIGRATION HELPER (MISSING CLASS)
# ========================================

class FixxyPromptMigrator:
    """
    Migration helper for FixxySidekick prompts (MISSING - Referenced in main_dynamic.py)
    
    This class handles migration of hardcoded prompts to dynamic storage.
    """
    
    def __init__(self, prompt_store: BasePromptStore):
        """Initialize migrator with storage backend"""
        self.prompt_store = prompt_store
        self.migration_id = f"migration_{int(datetime.utcnow().timestamp())}"
        logger.info(f"Initialized FixxyPromptMigrator ({self.migration_id})")
    
    async def migrate_hardcoded_prompts(self) -> Dict[str, Any]:
        """Migrate hardcoded prompts to dynamic storage"""
        migration_results = {
            "migration_id": self.migration_id,
            "start_time": datetime.utcnow().isoformat(),
            "prompts_migrated": 0,
            "migration_errors": [],
            "success": True
        }
        
        try:
            # Define some basic hardcoded prompts for migration
            hardcoded_prompts = [
                {
                    "task_type": "deduplicate",
                    "template": """You are an expert data analyst specializing in duplicate detection.

Task: Identify and remove duplicate records from the dataset.

Matching Fields: {match_fields}
Cleanup Rules: {cleanup_rules}
Data Context: {data_context}

Instructions:
1. Analyze records using specified matching fields
2. Apply fuzzy matching with similarity threshold
3. Keep the most complete record from each duplicate group
4. Provide summary of duplicates found and removed

Output: Cleaned dataset with duplicate removal report.""",
                    "input_variables": ["match_fields", "cleanup_rules", "data_context"],
                    "context": {
                        "field_list": ["email", "phone", "name"],
                        "cleanup_rules": {"fuzzy_matching": True, "similarity_threshold": 0.9},
                        "priority": "high"
                    }
                },
                {
                    "task_type": "format_cleanup",
                    "template": """You are an expert data standardization specialist.

Task: Standardize data formats according to specified rules.

Format Rules: {format_rules}
Target Fields: {field_list}
Data Context: {data_context}

Instructions:
1. Apply format rules to all specified fields
2. Convert dates to standardized format
3. Standardize phone numbers to international format
4. Report any format conversion issues

Output: Standardized dataset with format conversion summary.""",
                    "input_variables": ["format_rules", "field_list", "data_context"],
                    "context": {
                        "field_list": ["date", "phone", "currency"],
                        "format_rules": {"date_format": "ISO-8601", "phone_format": "international"},
                        "priority": "medium"
                    }
                }
            ]
            
            # Migrate each prompt
            for prompt_data in hardcoded_prompts:
                try:
                    # Create context signature and hash
                    context_hash = generate_context_hash(prompt_data["context"])
                    context_signature = create_context_signature_from_dict(prompt_data["context"])
                    
                    # Create stored prompt
                    prompt_id = f"fixxy_{prompt_data['task_type']}_legacy_{context_hash[:8]}"
                    
                    stored_prompt = StoredPrompt(
                        prompt_id=prompt_id,
                        sidekick_name="fixxy",
                        task_type=prompt_data["task_type"],
                        context_hash=context_hash,
                        prompt_template=prompt_data["template"],
                        input_variables=prompt_data["input_variables"],
                        context_signature=context_signature.dict(exclude_none=True),
                        performance_metrics={"migrated_from": "hardcoded", "migration_id": self.migration_id},
                        created_at=datetime.utcnow(),
                        last_used=datetime.utcnow(),
                        usage_count=0,
                        success_rate=0.85,  # Assume good performance for legacy
                        avg_response_time=2.5,
                        version="1.0",
                        status=PromptStatus.ACTIVE,
                        tags=["fixxy", prompt_data["task_type"], "legacy_migration"]
                    )
                    
                    # Save to storage
                    await self.prompt_store.save_prompt(stored_prompt)
                    migration_results["prompts_migrated"] += 1
                    
                    logger.info(f"Migrated prompt for {prompt_data['task_type']}")
                    
                except Exception as e:
                    error_msg = f"Failed to migrate {prompt_data['task_type']}: {str(e)}"
                    migration_results["migration_errors"].append(error_msg)
                    logger.error(error_msg)
            
            migration_results["end_time"] = datetime.utcnow().isoformat()
            migration_results["success"] = len(migration_results["migration_errors"]) == 0
            
            logger.info(f"Migration completed: {migration_results['prompts_migrated']} prompts migrated")
            return migration_results
            
        except Exception as e:
            migration_results["fatal_error"] = str(e)
            migration_results["success"] = False
            logger.error(f"Migration failed: {str(e)}")
            return migration_results

# ========================================
# FACTORY FUNCTION
# ========================================

def create_prompt_store(storage_type: str = "memory", **kwargs) -> BasePromptStore:
    """
    Factory function to create appropriate prompt store
    
    Args:
        storage_type: "memory" or "cosmos"
        **kwargs: Configuration for the storage type
        
    Returns:
        Configured prompt store instance
    """
    if storage_type == "cosmos":
        connection_string = kwargs.get("connection_string")
        if not connection_string:
            raise ValueError("connection_string required for cosmos storage")
        return CosmosFixxyPromptStore(connection_string)
    else:
        return FixxyPromptStore()

# ========================================
# EXPORTS
# ========================================

__all__ = [
    "BasePromptStore",
    "FixxyPromptStore",
    "CosmosFixxyPromptStore", 
    "FixxyPromptMigrator",
    "create_prompt_store"
]