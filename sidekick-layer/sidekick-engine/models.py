"""
Sidekick Engine Data Models
Fixed for Azure Functions deployment with proper validation
"""

import json
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

# Use Pydantic if available, otherwise fallback to manual validation
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Simple BaseModel fallback
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

# --------------------
# Request Models
# --------------------

class SidekickEngineRequest:
    """Request model for Sidekick Engine"""
    
    def __init__(self, **data):
        # Required fields
        self.request_id: str = data.get('request_id', f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        self.job_id: str = data.get('job_id', f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        self.task_type: str = data.get('task_type', 'unknown')
        self.sidekick_name: str = data.get('sidekick_name', 'fixxy')
        
        # Optional fields
        self.requested_version: Optional[str] = data.get('requested_version')
        self.ab_test_group: Optional[str] = data.get('ab_test_group')
        self.job_context: Dict[str, Any] = data.get('job_context', {})
        self.caller_component: str = data.get('caller_component', 'sidekick-router')
        self.timestamp_requested: str = data.get('timestamp_requested', datetime.utcnow().isoformat())
        self.schema_version: str = data.get('schema_version', 'v1')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'request_id': self.request_id,
            'job_id': self.job_id,
            'task_type': self.task_type,
            'sidekick_name': self.sidekick_name,
            'requested_version': self.requested_version,
            'ab_test_group': self.ab_test_group,
            'job_context': self.job_context,
            'caller_component': self.caller_component,
            'timestamp_requested': self.timestamp_requested,
            'schema_version': self.schema_version
        }

# --------------------
# Response Models  
# --------------------

class GeneratedPromptTemplate:
    """Generated prompt template model"""
    
    def __init__(self, **data):
        self.template_content: str = data.get('template_content', '')
        self.input_variables: List[str] = data.get('input_variables', [])
        self.template_format: str = data.get('template_format', 'standard')
        self.expected_output: str = data.get('expected_output', 'text')
        self.model_preference: str = data.get('model_preference', 'gpt-4')
        self.temperature: float = data.get('temperature', 0.7)
        self.max_tokens: int = data.get('max_tokens', 1000)
        self.stop_sequences: List[str] = data.get('stop_sequences', [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'template_content': self.template_content,
            'input_variables': self.input_variables,
            'template_format': self.template_format,
            'expected_output': self.expected_output,
            'model_preference': self.model_preference,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stop_sequences': self.stop_sequences
        }

class PromptTemplateMetadata:
    """Metadata about prompt generation"""
    
    def __init__(self, **data):
        self.sidekick_used: str = data.get('sidekick_used', 'unknown')
        self.generation_method: str = data.get('generation_method', 'template')
        self.llm_used: bool = data.get('llm_used', False)
        self.llm_model: Optional[str] = data.get('llm_model')
        self.llm_response_time: Optional[float] = data.get('llm_response_time')
        self.cache_hit: bool = data.get('cache_hit', False)
        self.template_version: str = data.get('template_version', 'v1.0')
        self.confidence_score: Optional[float] = data.get('confidence_score')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'sidekick_used': self.sidekick_used,
            'generation_method': self.generation_method,
            'llm_used': self.llm_used,
            'llm_model': self.llm_model,
            'llm_response_time': self.llm_response_time,
            'cache_hit': self.cache_hit,
            'template_version': self.template_version,
            'confidence_score': self.confidence_score
        }

class SidekickEngineResponse:
    """Response model for Sidekick Engine"""
    
    def __init__(self, **data):
        # Request traceability
        self.request_id: str = data.get('request_id', '')
        self.job_id: str = data.get('job_id', '')
        
        # Response status
        self.status: str = data.get('status', 'success')  # success, error, fallback
        
        # Core content
        self.prompt_template: Optional[GeneratedPromptTemplate] = data.get('prompt_template')
        
        # Error handling
        self.error_code: Optional[str] = data.get('error_code')
        self.error_message: Optional[str] = data.get('error_message')
        
        # Response metadata
        self.processed_at: str = data.get('processed_at', datetime.utcnow().isoformat())
        self.processing_time_ms: float = data.get('processing_time_ms', 0.0)
        self.engine_version: str = data.get('engine_version', 'v1.0')
        self.fallback_used: Optional[str] = data.get('fallback_used')
        self.metadata: Optional[PromptTemplateMetadata] = data.get('metadata')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'request_id': self.request_id,
            'job_id': self.job_id,
            'status': self.status,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'processed_at': self.processed_at,
            'processing_time_ms': self.processing_time_ms,
            'engine_version': self.engine_version,
            'fallback_used': self.fallback_used
        }
        
        if self.prompt_template:
            result['prompt_template'] = self.prompt_template.to_dict()
        
        if self.metadata:
            result['metadata'] = self.metadata.to_dict()
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    def model_dump_json(self) -> str:
        """Pydantic compatibility method"""
        return self.to_json()

# --------------------
# Error Models
# --------------------

class SidekickEngineError:
    """Error model for Sidekick Engine"""
    
    def __init__(self, **data):
        self.error_code: str = data.get('error_code', 'UNKNOWN_ERROR')
        self.error_message: str = data.get('error_message', 'An error occurred')
        self.request_id: Optional[str] = data.get('request_id')
        self.job_id: Optional[str] = data.get('job_id')
        self.timestamp: str = data.get('timestamp', datetime.utcnow().isoformat())
        self.component: str = data.get('component', 'sidekick-engine')
        self.sidekick_name: Optional[str] = data.get('sidekick_name')
        self.task_type: Optional[str] = data.get('task_type')
        self.error_details: Optional[Dict[str, Any]] = data.get('error_details')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'error_code': self.error_code,
            'error_message': self.error_message,
            'request_id': self.request_id,
            'job_id': self.job_id,
            'timestamp': self.timestamp,
            'component': self.component,
            'sidekick_name': self.sidekick_name,
            'task_type': self.task_type,
            'error_details': self.error_details
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

# --------------------
# Dynamic Prompt Models (for FixxySidekick)
# --------------------

class StoredPrompt:
    """Model for storing prompts in database with comprehensive validation"""
    
    def __init__(self, **data):
        # Validate and set required fields with safe defaults
        self.id: str = str(data.get('id', f"prompt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"))
        self.sidekick_name: str = str(data.get('sidekick_name', '')).strip()
        self.task_type: str = str(data.get('task_type', '')).strip()
        self.context_hash: str = str(data.get('context_hash', '')).strip()
        self.prompt_content: str = str(data.get('prompt_content', '')).strip()
        
        # Validate list fields with safe conversion
        input_vars = data.get('input_variables', [])
        if isinstance(input_vars, list):
            self.input_variables: List[str] = [str(x) for x in input_vars if x]
        else:
            self.input_variables: List[str] = []
        
        # Validate metadata with safe conversion
        metadata = data.get('metadata', {})
        if isinstance(metadata, dict):
            self.metadata: Dict[str, Any] = dict(metadata)
        else:
            self.metadata: Dict[str, Any] = {}
        
        # Validate and set timestamp fields
        try:
            self.created_at: str = str(data.get('created_at', datetime.utcnow().isoformat()))
            self.updated_at: str = str(data.get('updated_at', datetime.utcnow().isoformat()))
        except Exception:
            self.created_at = datetime.utcnow().isoformat()
            self.updated_at = datetime.utcnow().isoformat()
        
        # Validate and set numeric fields with safe conversion
        try:
            self.usage_count: int = max(0, int(data.get('usage_count', 0)))
            self.success_rate: float = max(0.0, min(1.0, float(data.get('success_rate', 0.0))))
            self.avg_response_time: float = max(0.0, float(data.get('avg_response_time', 0.0)))
        except (ValueError, TypeError):
            self.usage_count = 0
            self.success_rate = 0.0
            self.avg_response_time = 0.0
        
        # Validate status with safe default
        status = str(data.get('status', 'active')).lower().strip()
        if status in ['active', 'deprecated', 'testing', 'disabled']:
            self.status = status
        else:
            self.status = 'active'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage with error handling"""
        try:
            return {
                'id': self.id,
                'sidekick_name': self.sidekick_name,
                'task_type': self.task_type,
                'context_hash': self.context_hash,
                'prompt_content': self.prompt_content,
                'input_variables': list(self.input_variables),  # Ensure it's a new list
                'metadata': dict(self.metadata),  # Ensure it's a new dict
                'created_at': self.created_at,
                'updated_at': self.updated_at,
                'usage_count': self.usage_count,
                'success_rate': self.success_rate,
                'avg_response_time': self.avg_response_time,
                'status': self.status
            }
        except Exception as e:
            # Return minimal safe dictionary on error
            return {
                'id': str(getattr(self, 'id', 'error')),
                'sidekick_name': str(getattr(self, 'sidekick_name', '')),
                'task_type': str(getattr(self, 'task_type', '')),
                'status': 'error',
                'error': str(e)
            }
    
    def __eq__(self, other):
        """Equality comparison for caching"""
        if not isinstance(other, StoredPrompt):
            return False
        return self.id == other.id
    
    def __hash__(self):
        """Hash implementation for caching"""
        return hash(self.id)

# --------------------
# Additional Models for Storage (MISSING - CRITICAL FIX)
# --------------------

class ContextSignature:
    """Context signature for similarity matching"""
    
    def __init__(self, **data):
        self.field_count = data.get('field_count', 0)
        self.has_nested = data.get('has_nested', False)
        self.signature_data = data.get('signature_data', {})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_count': self.field_count,
            'has_nested': self.has_nested,
            'signature_data': self.signature_data
        }

class StoredPrompt:
    """Model for storing prompts in database"""
    
    def __init__(self, **data):
        self.id: str = data.get('id', f"prompt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        self.sidekick_name: str = data.get('sidekick_name', '')
        self.task_type: str = data.get('task_type', '')
        self.context_hash: str = data.get('context_hash', '')
        self.prompt_content: str = data.get('prompt_content', '')
        self.input_variables: List[str] = data.get('input_variables', [])
        self.metadata: Dict[str, Any] = data.get('metadata', {})
        self.created_at: str = data.get('created_at', datetime.utcnow().isoformat())
        self.updated_at: str = data.get('updated_at', datetime.utcnow().isoformat())
        self.usage_count: int = data.get('usage_count', 0)
        self.success_rate: float = data.get('success_rate', 0.0)
        self.avg_response_time: float = data.get('avg_response_time', 0.0)
        self.status: str = data.get('status', 'active')  # active, deprecated, testing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'id': self.id,
            'sidekick_name': self.sidekick_name,
            'task_type': self.task_type,
            'context_hash': self.context_hash,
            'prompt_content': self.prompt_content,
            'input_variables': self.input_variables,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
            'avg_response_time': self.avg_response_time,
            'status': self.status
        }

class PromptStatus:
    """Enumeration for prompt status values"""
    ACTIVE = "active"
    DEPRECATED = "deprecated" 
    TESTING = "testing"
    DISABLED = "disabled"

class ContextSignature:
    """Context signature for similarity matching"""
    
    def __init__(self, **data):
        self.field_count = data.get('field_count', 0)
        self.has_nested = data.get('has_nested', False)
        self.signature_data = data.get('signature_data', {})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_count': self.field_count,
            'has_nested': self.has_nested,
            'signature_data': self.signature_data
        }

def create_context_signature_from_dict(data: Dict[str, Any]) -> ContextSignature:
    """Create context signature from dictionary"""
    return ContextSignature(**data)

def generate_context_hash(context: Dict[str, Any]) -> str:
    """Generate hash for context"""
    import hashlib
    import json
    try:
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(context_str.encode()).hexdigest()
    except Exception:
        return hashlib.md5(str(context).encode()).hexdigest()

# --------------------
# Validation Helper
# --------------------

def validate_request(data: Dict[str, Any]) -> SidekickEngineRequest:
    """Validate and create SidekickEngineRequest with comprehensive error handling"""
    
    if not isinstance(data, dict):
        raise ValidationError("Request data must be a dictionary")
    
    # Check required fields with safe access
    required_fields = ['job_id', 'sidekick_name', 'task_type']
    
    for field in required_fields:
        if field not in data or not data.get(field):
            raise ValidationError(f"Missing or empty required field: {field}")
    
    # Validate field types and values with safe access
    job_id = data.get('job_id')
    if not isinstance(job_id, str) or len(job_id.strip()) == 0:
        raise ValidationError("job_id must be a non-empty string")
    
    sidekick_name = data.get('sidekick_name')
    if not isinstance(sidekick_name, str) or len(sidekick_name.strip()) == 0:
        raise ValidationError("sidekick_name must be a non-empty string")
        
    task_type = data.get('task_type')
    if not isinstance(task_type, str) or len(task_type.strip()) == 0:
        raise ValidationError("task_type must be a non-empty string")
    
    # Validate job_context if provided with safe access
    job_context = data.get('job_context')
    if job_context is not None and not isinstance(job_context, dict):
        raise ValidationError("job_context must be a dictionary if provided")
    
    # Validate optional fields with safe access
    requested_version = data.get('requested_version')
    if requested_version is not None and not isinstance(requested_version, str):
        raise ValidationError("requested_version must be a string if provided")
    
    # Create and return validated request with error handling
    try:
        return SidekickEngineRequest(**data)
    except Exception as e:
        raise ValidationError(f"Invalid request data: {str(e)}")

class ValidationError(Exception):
    """Custom validation error for request validation"""
    
    def __init__(self, message: str, errors: List[str] = None):
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)