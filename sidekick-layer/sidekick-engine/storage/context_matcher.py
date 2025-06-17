"""
File: /sidekick-engine/storage/context_matcher.py

Fixed context matching, similarity scoring, and hashing for dynamic prompt storage.

FIXES APPLIED:
- Fixed all import statements and typing
- Added proper error handling throughout
- Fixed model imports from models package
- Corrected method signatures and return types
- Added comprehensive logging
- Fixed ContextSignature handling
"""

import hashlib
import json
import logging
import re
from typing import Dict, Any, List, Set, Optional, Union

# Fixed imports - using absolute imports for models
try:
    from ..models.dynamic_prompt_models import ContextSignature
except ImportError:
    # Fallback for development/testing
    from models.dynamic_prompt_models import ContextSignature

logger = logging.getLogger(__name__)

class ContextMatcher:
    """Handles context matching and similarity scoring for prompt reuse"""
    
    # Define which context parameters are most important for each Sidekick type
    CRITICAL_PARAMS = {
        "fixxy": ["field_list", "data_types", "cleanup_rules", "validation_rules"],
        "findy": ["analysis_focus", "data_types", "field_list", "interest_areas"],
        "predicty": ["forecast_horizon", "target_variables", "confidence_level", "seasonal_factors"]
    }
    
    # Weight different parameter types for similarity calculation
    PARAM_WEIGHTS = {
        "field_list": 0.3,          # High weight - fields are critical
        "data_types": 0.25,         # High weight - data types matter
        "cleanup_rules": 0.2,       # Medium weight - affects approach
        "validation_rules": 0.2,    # Medium weight - affects validation
        "analysis_focus": 0.3,      # High weight for analysis tasks
        "forecast_horizon": 0.25,   # High weight for forecasting
        "confidence_level": 0.15,   # Medium weight
        "priority": 0.1,            # Low weight - doesn't affect core logic
        "seasonal_factors": 0.2,    # Medium weight for forecasting
        "external_factors": 0.15    # Medium weight
    }
    
    @staticmethod
    def create_context_hash(job_context: Dict[str, Any]) -> str:
        """
        Create a deterministic hash of the job context for exact matching.
        
        Args:
            job_context: The job context dictionary
            
        Returns:
            str: A short hash string for exact matching
        """
        try:
            # Validate input
            if not isinstance(job_context, dict):
                logger.warning(f"Invalid job_context type: {type(job_context)}, using empty dict")
                job_context = {}
            
            # Normalize context for consistent hashing
            normalized = ContextMatcher._normalize_context(job_context)
            
            # Create deterministic string representation
            context_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
            
            # Generate hash
            hash_obj = hashlib.sha256(context_str.encode('utf-8'))
            return hash_obj.hexdigest()[:16]  # Use first 16 characters
            
        except Exception as e:
            logger.error(f"Error creating context hash: {str(e)}")
            # Fallback to a simple hash
            try:
                fallback_str = str(sorted(job_context.items())) if job_context else "empty_context"
                return hashlib.md5(fallback_str.encode()).hexdigest()[:16]
            except Exception:
                # Ultimate fallback
                return hashlib.md5("fallback_context".encode()).hexdigest()[:16]
    
    @staticmethod
    def create_context_signature(job_context: Dict[str, Any], sidekick_name: Optional[str] = None) -> ContextSignature:
        """
        Create a normalized context signature for similarity matching.
        
        Args:
            job_context: The job context dictionary
            sidekick_name: Name of the Sidekick (for parameter prioritization)
            
        Returns:
            ContextSignature: Normalized signature for similarity comparison
        """
        try:
            # Validate input
            if not isinstance(job_context, dict):
                logger.warning(f"Invalid job_context type: {type(job_context)}, using empty dict")
                job_context = {}
            
            signature_data = {}
            
            # Extract and normalize key parameters
            key_params = [
                "field_list", "data_types", "analysis_focus", "priority",
                "validation_rules", "cleanup_rules", "format_rules",
                "forecast_horizon", "confidence_level", "seasonal_factors", "external_factors"
            ]
            
            for param in key_params:
                if param in job_context:
                    value = job_context[param]
                    normalized_value = ContextMatcher._normalize_parameter(param, value)
                    if normalized_value:  # Only add non-empty values
                        signature_data[param] = normalized_value
            
            # Convert to ContextSignature format
            context_signature = ContextSignature(
                field_list=ContextMatcher._extract_field_list(signature_data),
                data_types=ContextMatcher._extract_data_types(signature_data),
                task_parameters=ContextMatcher._extract_task_parameters(signature_data),
                schema_hints=ContextMatcher._extract_schema_hints(signature_data)
            )
            
            return context_signature
            
        except Exception as e:
            logger.error(f"Error creating context signature: {str(e)}")
            # Return empty signature as fallback
            return ContextSignature()
    
    @staticmethod
    def _extract_field_list(signature_data: Dict[str, Any]) -> List[str]:
        """Extract field list from signature data"""
        field_list = []
        
        if "field_list" in signature_data:
            raw_fields = signature_data["field_list"]
            if isinstance(raw_fields, str):
                field_list = [f.strip() for f in raw_fields.split(',') if f.strip()]
            elif isinstance(raw_fields, list):
                field_list = [str(f).strip() for f in raw_fields if f]
        
        return field_list
    
    @staticmethod
    def _extract_data_types(signature_data: Dict[str, Any]) -> List[str]:
        """Extract data types from signature data"""
        data_types = []
        
        if "data_types" in signature_data:
            raw_types = signature_data["data_types"]
            if isinstance(raw_types, str):
                data_types = [t.strip() for t in raw_types.split(',') if t.strip()]
            elif isinstance(raw_types, list):
                data_types = [str(t).strip() for t in raw_types if t]
        
        return data_types
    
    @staticmethod
    def _extract_task_parameters(signature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task parameters from signature data"""
        task_params = {}
        
        param_keys = [
            "cleanup_rules", "validation_rules", "format_rules",
            "forecast_horizon", "confidence_level", "priority"
        ]
        
        for key in param_keys:
            if key in signature_data:
                task_params[key] = signature_data[key]
        
        return task_params
    
    @staticmethod
    def _extract_schema_hints(signature_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract schema hints from signature data"""
        schema_hints = {}
        
        hint_keys = ["analysis_focus", "seasonal_factors", "external_factors"]
        
        for key in hint_keys:
            if key in signature_data:
                value = signature_data[key]
                schema_hints[key] = str(value) if value is not None else ""
        
        return schema_hints
    
    @staticmethod
    def calculate_similarity(signature1: ContextSignature, signature2: ContextSignature, 
                           sidekick_name: Optional[str] = None) -> float:
        """
        Calculate similarity between two context signatures.
        
        Args:
            signature1: First context signature
            signature2: Second context signature
            sidekick_name: Name of Sidekick for weighted comparison
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        try:
            if not signature1 or not signature2:
                return 0.0
            
            # Convert signatures to dictionaries for comparison
            try:
                sig1_dict = signature1.model_dump(exclude_none=True)
                sig2_dict = signature2.model_dump(exclude_none=True)
            except AttributeError:
                # Fallback for older Pydantic versions
                sig1_dict = signature1.dict(exclude_none=True)
                sig2_dict = signature2.dict(exclude_none=True)
            
            if not sig1_dict and not sig2_dict:
                return 1.0
            
            # Get critical parameters for this Sidekick
            critical_params = ContextMatcher.CRITICAL_PARAMS.get(sidekick_name or "", [])
            
            total_weight = 0.0
            weighted_matches = 0.0
            
            # Get all unique parameters from both signatures
            all_params = set(sig1_dict.keys()) | set(sig2_dict.keys())
            
            for param in all_params:
                # Get parameter weight
                weight = ContextMatcher.PARAM_WEIGHTS.get(param, 0.1)
                
                # Increase weight for critical parameters
                if param in critical_params:
                    weight *= 1.5
                
                total_weight += weight
                
                # Calculate parameter similarity
                param_similarity = ContextMatcher._compare_parameter_values(
                    sig1_dict.get(param), sig2_dict.get(param), param
                )
                
                weighted_matches += param_similarity * weight
            
            # Calculate final similarity score
            similarity = weighted_matches / total_weight if total_weight > 0 else 0.0
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def _normalize_context(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize context for consistent processing.
        
        Args:
            context: Raw context dictionary
            
        Returns:
            Dict: Normalized context
        """
        normalized = {}
        
        try:
            for key, value in context.items():
                if value is None:
                    continue
                    
                if isinstance(value, list):
                    # Sort lists for consistency, handle mixed types
                    try:
                        normalized[key] = sorted([str(item) for item in value if item is not None])
                    except Exception:
                        normalized[key] = [str(item) for item in value if item is not None]
                elif isinstance(value, dict):
                    # Sort dict items for consistency
                    try:
                        normalized[key] = dict(sorted(value.items()))
                    except Exception:
                        normalized[key] = value
                elif isinstance(value, (int, float, bool)):
                    # Keep numeric and boolean values as-is
                    normalized[key] = value
                else:
                    # Convert to string and normalize case
                    normalized[key] = str(value).lower().strip()
        except Exception as e:
            logger.error(f"Error normalizing context: {str(e)}")
            
        return normalized
    
    @staticmethod
    def _normalize_parameter(param_name: str, value: Any) -> str:
        """
        Normalize a specific parameter value for consistent comparison.
        
        Args:
            param_name: Name of the parameter
            value: Value to normalize
            
        Returns:
            str: Normalized string representation
        """
        try:
            if value is None:
                return ""
            
            if isinstance(value, list):
                # Sort and join list items
                try:
                    sorted_items = sorted([str(item).lower().strip() for item in value if item is not None])
                    return ",".join(sorted_items)
                except Exception:
                    return ",".join([str(item).lower().strip() for item in value if item is not None])
            elif isinstance(value, dict):
                # Sort and format dict items
                try:
                    sorted_items = sorted(f"{k}:{v}" for k, v in value.items() if v is not None)
                    return ",".join(sorted_items)
                except Exception:
                    return str(value).lower().strip()
            else:
                # Convert to lowercase string
                return str(value).lower().strip()
        except Exception as e:
            logger.warning(f"Error normalizing parameter {param_name}: {str(e)}")
            return ""
    
    @staticmethod
    def _compare_parameter_values(value1: Any, value2: Any, param_name: str) -> float:
        """
        Compare two parameter values and return similarity score.
        
        Args:
            value1: First value
            value2: Second value
            param_name: Name of the parameter being compared
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        try:
            # Handle None values
            if value1 is None and value2 is None:
                return 1.0
            if value1 is None or value2 is None:
                return 0.0
            
            # Convert to strings for comparison
            str1 = str(value1).lower().strip()
            str2 = str(value2).lower().strip()
            
            # Exact match
            if str1 == str2:
                return 1.0
            
            # For list-like parameters, use Jaccard similarity
            if param_name in ["field_list", "seasonal_factors", "external_factors"]:
                return ContextMatcher._jaccard_similarity(str1, str2)
            
            # For numeric parameters, use range-based similarity
            if param_name in ["confidence_level", "similarity_threshold"]:
                return ContextMatcher._numeric_similarity(str1, str2)
            
            # For categorical parameters, partial match
            if param_name in ["priority", "analysis_focus", "data_types"]:
                return ContextMatcher._categorical_similarity(str1, str2)
            
            # Default: substring similarity
            return ContextMatcher._substring_similarity(str1, str2)
            
        except Exception as e:
            logger.warning(f"Error comparing parameter {param_name}: {str(e)}")
            return 0.0
    
    @staticmethod
    def _jaccard_similarity(str1: str, str2: str) -> float:
        """Calculate Jaccard similarity for comma-separated values"""
        try:
            set1 = set(str1.split(',')) if str1 else set()
            set2 = set(str2.split(',')) if str2 else set()
            
            if not set1 and not set2:
                return 1.0
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def _numeric_similarity(str1: str, str2: str) -> float:
        """Calculate similarity for numeric values"""
        try:
            # Try to extract numeric values
            num1_match = re.search(r'[\d.]+', str1)
            num2_match = re.search(r'[\d.]+', str2)
            
            if not num1_match or not num2_match:
                return 1.0 if str1 == str2 else 0.0
            
            num1 = float(num1_match.group())
            num2 = float(num2_match.group())
            
            # Calculate relative difference
            if num1 == 0 and num2 == 0:
                return 1.0
            
            max_val = max(abs(num1), abs(num2))
            if max_val == 0:
                return 1.0
            
            diff = abs(num1 - num2) / max_val
            return max(0.0, 1.0 - diff)
            
        except (ValueError, AttributeError):
            # Fallback to string comparison
            return 1.0 if str1 == str2 else 0.0
    
    @staticmethod
    def _categorical_similarity(str1: str, str2: str) -> float:
        """Calculate similarity for categorical values"""
        try:
            # Check for partial matches in categorical values
            if str1 in str2 or str2 in str1:
                return 0.7  # Partial match
            
            # Check for common keywords
            words1 = set(str1.split())
            words2 = set(str2.split())
            
            if words1 & words2:  # Common words exist
                return 0.5
            
            return 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def _substring_similarity(str1: str, str2: str) -> float:
        """Calculate similarity based on longest common substring"""
        try:
            if len(str1) == 0 and len(str2) == 0:
                return 1.0
            if len(str1) == 0 or len(str2) == 0:
                return 0.0
            
            # Simple Levenshtein-like similarity
            longer = str1 if len(str1) > len(str2) else str2
            shorter = str2 if len(str1) > len(str2) else str1
            
            if longer == shorter:
                return 1.0
            
            # Count common characters
            common_chars = sum(1 for c in shorter if c in longer)
            return common_chars / len(longer)
        except Exception:
            return 0.0
    
    @staticmethod
    def is_context_significant_change(old_signature: ContextSignature, 
                                    new_signature: ContextSignature,
                                    sidekick_name: Optional[str] = None) -> bool:
        """
        Determine if context change is significant enough to warrant new prompt generation.
        
        Args:
            old_signature: Previous context signature
            new_signature: New context signature
            sidekick_name: Name of Sidekick for weighted comparison
            
        Returns:
            bool: True if change is significant (requires new prompt)
        """
        try:
            similarity = ContextMatcher.calculate_similarity(old_signature, new_signature, sidekick_name)
            
            # Threshold for significant change (lower similarity means more change)
            significant_change_threshold = 0.6
            
            return similarity < significant_change_threshold
        except Exception as e:
            logger.error(f"Error checking context change significance: {str(e)}")
            # Default to significant change for safety
            return True