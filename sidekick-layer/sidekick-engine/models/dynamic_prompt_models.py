"""
dynamic_prompt_models.py

Fixed Pydantic data models for the dynamic prompt storage system.

FIXES APPLIED:
- Added missing PromptSource and PromptStatus enums
- Fixed all import statements
- Added proper type hints throughout
- Fixed PromptMigrationRecord model
- Corrected validation methods for Pydantic v2
- Added missing utility functions
- Fixed datetime handling
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from enum import Enum
import hashlib
import json

class PromptSource(Enum):
    """How a prompt was obtained"""
    DATABASE = "database"
    GENERATED = "generated"
    FALLBACK = "fallback"
    SIMILAR_MATCH = "similar_match"
    EXACT_MATCH = "exact_match"

class PromptStatus(Enum):
    """Status of a stored prompt"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    TESTING = "testing"

class ContextSignature(BaseModel):
    """
    Represents the context pattern of a prompt request.
    Used to match similar requests and enable prompt reuse.
    """
    field_list: List[str] = Field(default_factory=list, description="List of data fields involved")
    data_types: List[str] = Field(default_factory=list, description="Types of data being processed")
    task_parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional task parameters")
    schema_hints: Dict[str, str] = Field(default_factory=dict, description="Data schema information")
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }
    
    def normalize(self) -> "ContextSignature":
        """
        Normalize the context signature for consistent comparison.
        
        Returns:
            Normalized version with sorted lists and lowercased strings
        """
        return ContextSignature(
            field_list=sorted([field.lower().strip() for field in self.field_list]),
            data_types=sorted([dtype.lower().strip() for dtype in self.data_types]),
            task_parameters=self.task_parameters,
            schema_hints={k.lower(): v.lower() for k, v in self.schema_hints.items()}
        )
    
    def to_hash_string(self) -> str:
        """
        Convert context signature to a string suitable for hashing.
        
        Returns:
            Deterministic string representation
        """
        normalized = self.normalize()
        hash_data = {
            "fields": normalized.field_list,
            "types": normalized.data_types,
            "params": sorted(normalized.task_parameters.items()) if normalized.task_parameters else [],
            "schema": sorted(normalized.schema_hints.items()) if normalized.schema_hints else []
        }
        return json.dumps(hash_data, sort_keys=True, separators=(',', ':'))


class PromptMetadata(BaseModel):
    """
    Metadata about prompt usage and performance.
    """
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate (0.0 to 1.0)")
    usage_count: int = Field(default=0, ge=0, description="Number of times this prompt has been used")
    average_execution_time: Optional[float] = Field(default=None, description="Average execution time in seconds")
    last_success: Optional[str] = Field(default=None, description="ISO timestamp of last successful use")
    last_failure: Optional[str] = Field(default=None, description="ISO timestamp of last failure")
    failure_count: int = Field(default=0, ge=0, description="Number of times this prompt failed")
    
    @field_validator('success_rate')
    @classmethod
    def validate_success_rate(cls, v: float) -> float:
        """Ensure success rate is between 0 and 1"""
        return max(0.0, min(1.0, v))
    
    def update_success(self, execution_time: Optional[float] = None) -> None:
        """
        Update metadata after a successful prompt execution.
        
        Args:
            execution_time: Time taken for execution in seconds
        """
        self.usage_count += 1
        successful_uses = int(self.success_rate * (self.usage_count - 1))
        self.success_rate = (successful_uses + 1) / self.usage_count
        self.last_success = datetime.utcnow().isoformat()
        
        if execution_time is not None:
            if self.average_execution_time is None:
                self.average_execution_time = execution_time
            else:
                # Weighted average with more weight on recent executions
                self.average_execution_time = (self.average_execution_time * 0.8) + (execution_time * 0.2)
    
    def update_failure(self) -> None:
        """Update metadata after a failed prompt execution."""
        self.usage_count += 1
        self.failure_count += 1
        successful_uses = int(self.success_rate * (self.usage_count - 1))
        self.success_rate = successful_uses / self.usage_count if self.usage_count > 0 else 0.0
        self.last_failure = datetime.utcnow().isoformat()


class StoredPrompt(BaseModel):
    """
    Main model for storing prompts in the dynamic prompt system.
    
    This represents a complete prompt template that can be retrieved
    and reused for similar tasks.
    """
    # Core identification
    prompt_id: str = Field(..., description="Unique identifier for this prompt")
    sidekick_name: str = Field(..., description="Name of the sidekick that created this prompt")
    task_type: str = Field(..., description="Type of task this prompt handles")
    
    # Content and context
    prompt_template: str = Field(..., description="The actual prompt template text")
    context_hash: str = Field(..., description="Hash of the context signature")
    context_signature: Dict[str, Any] = Field(default_factory=dict, description="Original context that generated this prompt")
    input_variables: List[str] = Field(default_factory=list, description="Variables needed for the template")
    
    # Metadata and performance
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance tracking data")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_used: datetime = Field(default_factory=datetime.utcnow, description="Last usage timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    # Performance tracking
    usage_count: int = Field(default=0, description="Times this prompt has been used")
    success_rate: float = Field(default=0.5, description="Success rate of this prompt")
    avg_response_time: Optional[float] = Field(default=None, description="Average response time")
    
    # Optional categorization
    tags: List[str] = Field(default_factory=list, description="Optional tags for categorization")
    version: str = Field(default="1.0", description="Version of this prompt")
    status: PromptStatus = Field(default=PromptStatus.ACTIVE, description="Status of this prompt")
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat(),
            PromptStatus: lambda status: status.value
        }
    }
    
    @field_validator('prompt_id')
    @classmethod
    def validate_prompt_id(cls, v: str) -> str:
        """Ensure prompt ID follows expected format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Prompt ID cannot be empty")
        return v.strip().lower()
    
    @field_validator('sidekick_name')
    @classmethod
    def validate_sidekick_name(cls, v: str) -> str:
        """Ensure sidekick name is valid"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Sidekick name cannot be empty")
        return v.strip().lower()
    
    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Ensure task type is valid"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Task type cannot be empty")
        return v.strip().lower()
    
    @field_validator('prompt_template')
    @classmethod
    def validate_prompt_template(cls, v: str) -> str:
        """Ensure prompt template is not empty"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Prompt template cannot be empty")
        return v.strip()
    
    def update_usage(self, success: bool = True, execution_time: Optional[float] = None) -> None:
        """
        Update usage statistics for this prompt.
        
        Args:
            success: Whether the prompt execution was successful
            execution_time: Time taken for execution in seconds
        """
        self.usage_count += 1
        
        if success:
            successful_uses = int(self.success_rate * (self.usage_count - 1))
            self.success_rate = (successful_uses + 1) / self.usage_count
        else:
            successful_uses = int(self.success_rate * (self.usage_count - 1))
            self.success_rate = successful_uses / self.usage_count
        
        if execution_time is not None:
            if self.avg_response_time is None:
                self.avg_response_time = execution_time
            else:
                self.avg_response_time = (self.avg_response_time * 0.8) + (execution_time * 0.2)
        
        self.last_used = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def is_high_performing(self, min_usage: int = 5, min_success_rate: float = 0.8) -> bool:
        """
        Check if this prompt is considered high-performing.
        
        Args:
            min_usage: Minimum number of uses to consider
            min_success_rate: Minimum success rate to consider high-performing
            
        Returns:
            True if prompt meets high-performance criteria
        """
        return (
            self.usage_count >= min_usage and 
            self.success_rate >= min_success_rate
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for database storage.
        
        Returns:
            Dictionary representation suitable for CosmosDB
        """
        return {
            "id": self.prompt_id,  # CosmosDB requires 'id' field
            "prompt_id": self.prompt_id,
            "sidekick_name": self.sidekick_name,
            "task_type": self.task_type,
            "prompt_template": self.prompt_template,
            "context_hash": self.context_hash,
            "context_signature": self.context_signature,
            "input_variables": self.input_variables,
            "performance_metrics": self.performance_metrics,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "tags": self.tags,
            "version": self.version,
            "status": self.status.value if isinstance(self.status, PromptStatus) else self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredPrompt":
        """
        Create StoredPrompt from dictionary (e.g., from database).
        
        Args:
            data: Dictionary from database
            
        Returns:
            StoredPrompt instance
        """
        # Handle CosmosDB 'id' field
        if 'id' in data and 'prompt_id' not in data:
            data['prompt_id'] = data['id']
        
        # Convert datetime strings back to datetime objects
        for field in ['created_at', 'last_used', 'updated_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert status string to enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = PromptStatus(data['status'])
        
        return cls(**data)


class PromptResponse(BaseModel):
    """
    Response model for prompt retrieval/generation.
    """
    prompt_template: str = Field(..., description="Generated prompt template")
    input_variables: List[str] = Field(default_factory=list, description="Variables needed for template")
    source: PromptSource = Field(..., description="How this prompt was obtained")
    prompt_id: Optional[str] = Field(None, description="ID if from database")
    context_match_score: float = Field(default=0.0, description="Similarity score (0.0-1.0)")
    performance_prediction: float = Field(default=0.5, description="Expected success rate")
    usage_count: int = Field(default=0, description="Times this prompt has been used")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat(),
            PromptSource: lambda source: source.value
        }
    }


class PromptGenerationRequest(BaseModel):
    """
    Request model for generating new prompts.
    """
    sidekick_name: str = Field(..., description="Name of the requesting sidekick")
    task_type: str = Field(..., description="Type of task to generate prompt for")
    job_context: Dict[str, Any] = Field(..., description="Context information for prompt generation")
    template_hints: Optional[List[str]] = Field(default=None, description="Hints for prompt structure")
    examples: Optional[List[Dict[str, Any]]] = Field(default=None, description="Example inputs/outputs")
    fallback_to_llm: bool = Field(default=True, description="Whether to use LLM if no match found")
    save_generated: bool = Field(default=True, description="Whether to save generated prompts")
    similarity_threshold: float = Field(default=0.8, description="Minimum similarity for reuse")
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }


class SimilarPrompt(BaseModel):
    """
    Represents a prompt that's similar to a requested context.
    Used for finding and adapting existing prompts.
    """
    stored_prompt: StoredPrompt = Field(..., description="The similar prompt found")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0)")
    matching_fields: List[str] = Field(default_factory=list, description="Fields that matched")
    differing_fields: List[str] = Field(default_factory=list, description="Fields that differed")
    adaptation_needed: bool = Field(default=False, description="Whether prompt needs adaptation")
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }
    
    def is_usable(self, min_similarity: float = 0.8) -> bool:
        """
        Check if this similar prompt is usable for the request.
        
        Args:
            min_similarity: Minimum similarity score to consider usable
            
        Returns:
            True if prompt is similar enough to use
        """
        return self.similarity_score >= min_similarity


class PromptMigrationRecord(BaseModel):
    """
    Record of prompt migration from legacy system.
    """
    migration_id: str = Field(..., description="Unique migration identifier")
    sidekick_name: str = Field(..., description="Name of the sidekick")
    task_type: str = Field(..., description="Type of task")
    original_template: str = Field(..., description="Original template (truncated)")
    migrated_prompt_id: str = Field(..., description="ID of migrated prompt")
    migration_date: datetime = Field(default_factory=datetime.utcnow, description="Migration timestamp")
    migration_status: str = Field(default="completed", description="Migration status")
    notes: Optional[str] = Field(default=None, description="Migration notes")
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }


# Utility functions for working with models

def generate_prompt_id(sidekick_name: str, task_type: str, context_hash: str) -> str:
    """
    Generate a unique prompt ID.
    
    Args:
        sidekick_name: Name of the sidekick
        task_type: Type of task
        context_hash: Hash of the context
        
    Returns:
        Unique prompt ID
    """
    return f"{sidekick_name.lower()}_{task_type.lower()}_{context_hash[:8]}"


def generate_context_hash(context_signature: ContextSignature) -> str:
    """
    Generate a hash for a context signature.
    
    Args:
        context_signature: The context to hash
        
    Returns:
        SHA-256 hash as hexadecimal string
    """
    hash_string = context_signature.to_hash_string()
    return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()


def create_context_signature(
    field_list: List[str],
    data_types: Optional[List[str]] = None,
    task_parameters: Optional[Dict[str, Any]] = None,
    schema_hints: Optional[Dict[str, str]] = None
) -> ContextSignature:
    """
    Create a context signature from components.
    
    Args:
        field_list: List of data fields
        data_types: List of data types
        task_parameters: Optional task parameters
        schema_hints: Optional schema hints
        
    Returns:
        ContextSignature instance
    """
    return ContextSignature(
        field_list=field_list or [],
        data_types=data_types or [],
        task_parameters=task_parameters or {},
        schema_hints=schema_hints or {}
    )


# Export all models and utilities
__all__ = [
    "ContextSignature",
    "PromptMetadata", 
    "StoredPrompt",
    "SimilarPrompt",
    "PromptGenerationRequest",
    "PromptResponse",
    "PromptMigrationRecord",
    "PromptSource",
    "PromptStatus",
    "generate_prompt_id",
    "generate_context_hash",
    "create_context_signature"
]