"""
FixxySidekick - Complete Data Cleanup Specialist Implementation
The fully functional data cleanup and standardization AI agent
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import from parent modules using absolute imports
try:
    from ..base_sidekick import BaseSidekick
    from ..models import GeneratedPromptTemplate, PromptTemplateMetadata, StoredPrompt
    from ..storage.cosmos_prompt_store import CosmosPromptStore
    from ..storage.dynamic_prompt_manager import DynamicPromptManager
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from base_sidekick import BaseSidekick
    from models import GeneratedPromptTemplate, PromptTemplateMetadata, StoredPrompt
    from storage.cosmos_prompt_store import CosmosPromptStore
    from storage.dynamic_prompt_manager import DynamicPromptManager

logger = logging.getLogger(__name__)

class FixxySidekick(BaseSidekick):
    """
    FixxySidekick - The Data Cleanup & Standardization Specialist
    
    Specializes in:
    - Data deduplication
    - Format standardization  
    - Data quality assessment
    - Field mapping and transformation
    - Data validation and cleansing
    """
    
    def __init__(self, prompt_manager: Optional[DynamicPromptManager] = None):
        """Initialize FixxySidekick with dynamic prompt capabilities"""
        
        super().__init__(
            name="fixxy",
            version="v2.0",
            display_name="Fixxy - Data Cleanup Specialist"
        )
        
        # Capabilities
        self._requires_llm = True
        self._supported_tasks = [
            "deduplicate", "standardize", "validate", "cleanse", 
            "format", "merge", "quality_check", "field_mapping"
        ]
        
        # Dynamic prompt management
        self.prompt_manager = prompt_manager
        
        # Task-specific configurations
        self._task_configs = {
            "deduplicate": {
                "model_preference": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 1500,
                "requires_llm": True
            },
            "standardize": {
                "model_preference": "gpt-3.5-turbo",
                "temperature": 0.2,
                "max_tokens": 1000,
                "requires_llm": False
            },
            "validate": {
                "model_preference": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 800,
                "requires_llm": True
            },
            "cleanse": {
                "model_preference": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": 1200,
                "requires_llm": True
            }
        }
        
        logger.info(f"[FIXXY] Initialized {self.display_name}")
    
    def _initialize_langchain_components(self):
        """Initialize LangChain components for FixxySidekick"""
        try:
            # Import LangChain components
            from langchain.prompts import PromptTemplate
            from langchain.schema import HumanMessage
            
            # Base prompt templates for each task type
            self._prompt_templates = {
                "deduplicate": PromptTemplate(
                    input_variables=["data_context", "field_list", "similarity_threshold"],
                    template=self._get_deduplicate_template()
                ),
                "standardize": PromptTemplate(
                    input_variables=["data_context", "target_format", "field_mappings"],
                    template=self._get_standardize_template()
                ),
                "validate": PromptTemplate(
                    input_variables=["data_context", "validation_rules", "field_requirements"],
                    template=self._get_validate_template()
                ),
                "cleanse": PromptTemplate(
                    input_variables=["data_context", "cleansing_rules", "quality_standards"],
                    template=self._get_cleanse_template()
                )
            }
            
            logger.info("[FIXXY] LangChain components initialized successfully")
            
        except ImportError as e:
            logger.warning(f"[FIXXY] LangChain not fully available: {e}")
            self._prompt_templates = {}
    
    async def generate_prompt_template(self, task_type: str, context: Dict[str, Any], 
                                     langchain_manager: Any, job_id: str) -> GeneratedPromptTemplate:
        """
        Generate a prompt template for data cleanup tasks
        
        This is the main method called by the Sidekick Engine to generate
        prompts for FixxySidekick's data cleanup capabilities.
        """
        
        logger.info(f"[{job_id}] FixxySidekick generating template for task: {task_type}")
        
        try:
            # Check if we have a stored prompt for this context
            if self.prompt_manager:
                stored_prompt = await self._try_get_stored_prompt(task_type, context, job_id)
                if stored_prompt:
                    logger.info(f"[{job_id}] Using stored prompt for {task_type}")
                    return stored_prompt
            
            # Generate new prompt based on task type
            if task_type in self._supported_tasks:
                template = await self._generate_task_specific_template(task_type, context, job_id)
                
                # Store the successful prompt for future use
                if self.prompt_manager and template:
                    await self._store_successful_prompt(task_type, context, template, job_id)
                
                return template
            else:
                # Fallback for unsupported tasks
                logger.warning(f"[{job_id}] Unsupported task type: {task_type}, using fallback")
                return self._create_fallback_template(task_type, context)
                
        except Exception as e:
            logger.error(f"[{job_id}] Error generating template: {str(e)}")
            return self._create_error_fallback_template(task_type, context, str(e))
    
    async def _generate_task_specific_template(self, task_type: str, context: Dict[str, Any], 
                                             job_id: str) -> GeneratedPromptTemplate:
        """Generate template specific to the task type"""
        
        if task_type == "deduplicate":
            return self._create_deduplicate_template(context, job_id)
        elif task_type == "standardize":
            return self._create_standardize_template(context, job_id)
        elif task_type == "validate":
            return self._create_validate_template(context, job_id)
        elif task_type == "cleanse":
            return self._create_cleanse_template(context, job_id)
        elif task_type == "format":
            return self._create_format_template(context, job_id)
        elif task_type == "merge":
            return self._create_merge_template(context, job_id)
        elif task_type == "quality_check":
            return self._create_quality_check_template(context, job_id)
        elif task_type == "field_mapping":
            return self._create_field_mapping_template(context, job_id)
        else:
            return self._create_fallback_template(task_type, context)
    
    def _create_deduplicate_template(self, context: Dict[str, Any], job_id: str) -> GeneratedPromptTemplate:
        """Create template for data deduplication"""
        
        field_list = context.get('field_list', ['id', 'email', 'name'])
        similarity_threshold = context.get('similarity_threshold', 0.85)
        
        template_content = f"""You are Fixxy, a data deduplication specialist. Your task is to identify and remove duplicate records from the provided dataset.

**TASK**: Remove duplicate records based on similarity analysis

**DEDUPLICATION CRITERIA**:
- Primary matching fields: {', '.join(field_list)}
- Similarity threshold: {similarity_threshold}
- Consider variations in formatting, case, and spacing

**INSTRUCTIONS**:
1. Analyze each record in the dataset
2. Group records that appear to be duplicates based on the matching fields
3. For each group of duplicates:
   - Keep the most complete record (least missing fields)
   - If completeness is equal, keep the most recent record
   - If recency is equal, keep the first occurrence
4. Provide a summary of duplicates found and removed

**INPUT VARIABLES**:
- data_context: {{data_context}}
- field_list: {{field_list}}
- similarity_threshold: {{similarity_threshold}}

**EXPECTED OUTPUT**:
Return a JSON object with:
- "deduplicated_data": Array of unique records
- "duplicates_removed": Number of duplicates removed
- "duplicate_groups": Array of duplicate groups found
- "summary": Text summary of the deduplication process

**QUALITY REQUIREMENTS**:
- Maintain data integrity
- Preserve the best version of each unique record
- Document all deduplication decisions
- Ensure no false positives in duplicate detection"""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "field_list", "similarity_threshold"],
            template_format="structured_analysis",
            expected_output="json_object",
            model_preference="gpt-4",
            temperature=0.1,
            max_tokens=1500,
            stop_sequences=[]
        )
    
    def _create_standardize_template(self, context: Dict[str, Any], job_id: str) -> GeneratedPromptTemplate:
        """Create template for data standardization"""
        
        target_format = context.get('target_format', 'standard')
        field_mappings = context.get('field_mappings', {})
        
        template_content = f"""You are Fixxy, a data standardization specialist. Your task is to standardize the format and structure of the provided dataset.

**TASK**: Standardize data format and structure

**STANDARDIZATION REQUIREMENTS**:
- Target format: {target_format}
- Field mappings: {json.dumps(field_mappings, indent=2)}
- Ensure consistent formatting across all records

**INSTRUCTIONS**:
1. Apply the specified field mappings to transform field names
2. Standardize data formats:
   - Dates: Convert to ISO 8601 format (YYYY-MM-DD)
   - Phone numbers: Format as +1-XXX-XXX-XXXX
   - Email addresses: Convert to lowercase
   - Names: Title case format
   - Addresses: Standardize abbreviations and formatting
3. Handle missing or invalid data appropriately
4. Maintain data relationships and integrity

**INPUT VARIABLES**:
- data_context: {{data_context}}
- target_format: {{target_format}}
- field_mappings: {{field_mappings}}

**EXPECTED OUTPUT**:
Return a JSON object with:
- "standardized_data": Array of standardized records
- "transformations_applied": List of transformations performed
- "validation_errors": Any data quality issues found
- "summary": Text summary of standardization process

**QUALITY REQUIREMENTS**:
- Consistent formatting across all records
- Preserve original data meaning
- Handle edge cases gracefully
- Document all transformations applied"""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "target_format", "field_mappings"],
            template_format="structured_transformation",
            expected_output="json_object",
            model_preference="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1000,
            stop_sequences=[]
        )
    
    def _create_validate_template(self, context: Dict[str, Any], job_id: str) -> GeneratedPromptTemplate:
        """Create template for data validation"""
        
        validation_rules = context.get('validation_rules', [])
        field_requirements = context.get('field_requirements', {})
        
        template_content = f"""You are Fixxy, a data validation specialist. Your task is to validate data quality and identify issues in the provided dataset.

**TASK**: Validate data quality and identify issues

**VALIDATION CRITERIA**:
- Validation rules: {json.dumps(validation_rules, indent=2)}
- Field requirements: {json.dumps(field_requirements, indent=2)}
- Check for completeness, accuracy, and consistency

**INSTRUCTIONS**:
1. Validate each record against the specified rules
2. Check for:
   - Missing required fields
   - Invalid data formats
   - Out-of-range values
   - Inconsistent data patterns
   - Referential integrity issues
3. Categorize issues by severity (critical, warning, info)
4. Provide specific recommendations for each issue

**INPUT VARIABLES**:
- data_context: {{data_context}}
- validation_rules: {{validation_rules}}
- field_requirements: {{field_requirements}}

**EXPECTED OUTPUT**:
Return a JSON object with:
- "validation_results": Array of validation results per record
- "issues_found": Categorized list of data quality issues
- "recommendations": Specific fixes for each issue type
- "overall_quality_score": Numeric score (0-100)
- "summary": Text summary of validation findings

**QUALITY REQUIREMENTS**:
- Thorough validation coverage
- Clear issue descriptions
- Actionable recommendations
- Accurate quality scoring"""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "validation_rules", "field_requirements"],
            template_format="quality_assessment",
            expected_output="json_object",
            model_preference="gpt-4",
            temperature=0.1,
            max_tokens=800,
            stop_sequences=[]
        )
    
    def _create_cleanse_template(self, context: Dict[str, Any], job_id: str) -> GeneratedPromptTemplate:
        """Create template for data cleansing"""
        
        cleansing_rules = context.get('cleansing_rules', [])
        quality_standards = context.get('quality_standards', {})
        
        template_content = f"""You are Fixxy, a data cleansing specialist. Your task is to clean and improve the quality of the provided dataset.

**TASK**: Clean and improve data quality

**CLEANSING CRITERIA**:
- Cleansing rules: {json.dumps(cleansing_rules, indent=2)}
- Quality standards: {json.dumps(quality_standards, indent=2)}
- Focus on accuracy, completeness, and consistency

**INSTRUCTIONS**:
1. Apply cleansing rules to each record
2. Perform these cleansing operations:
   - Remove or fix invalid characters
   - Correct obvious typos and misspellings
   - Standardize format variations
   - Fill missing values where appropriate
   - Remove or flag outlier values
3. Maintain an audit trail of all changes made
4. Preserve original values when corrections are uncertain

**INPUT VARIABLES**:
- data_context: {{data_context}}
- cleansing_rules: {{cleansing_rules}}
- quality_standards: {{quality_standards}}

**EXPECTED OUTPUT**:
Return a JSON object with:
- "cleansed_data": Array of cleaned records
- "changes_made": Detailed log of all modifications
- "quality_improvements": Before/after quality metrics
- "uncertain_corrections": Records requiring manual review
- "summary": Text summary of cleansing process

**QUALITY REQUIREMENTS**:
- Preserve data integrity
- Document all changes made
- Maintain traceability to original values
- Apply consistent cleansing standards"""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "cleansing_rules", "quality_standards"],
            template_format="data_cleaning",
            expected_output="json_object",
            model_preference="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=1200,
            stop_sequences=[]
        )
    
    def _create_format_template(self, context: Dict[str, Any], job_id: str) -> GeneratedPromptTemplate:
        """Create template for data formatting"""
        
        template_content = """You are Fixxy, a data formatting specialist. Transform data into the specified format while preserving all information.

**TASK**: Format data according to specifications

**INSTRUCTIONS**:
1. Apply the requested formatting rules
2. Ensure consistent structure across all records
3. Preserve data relationships and hierarchy
4. Handle special characters and encoding properly

**INPUT VARIABLES**:
- data_context: {data_context}
- format_specifications: {format_specifications}

**EXPECTED OUTPUT**:
Return formatted data maintaining original information integrity."""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "format_specifications"],
            template_format="data_formatting",
            expected_output="formatted_data",
            model_preference="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1000,
            stop_sequences=[]
        )
    
    def _create_merge_template(self, context: Dict[str, Any], job_id: str) -> GeneratedPromptTemplate:
        """Create template for data merging"""
        
        template_content = """You are Fixxy, a data merging specialist. Combine multiple datasets while maintaining data quality and relationships.

**TASK**: Merge datasets intelligently

**INSTRUCTIONS**:
1. Identify matching records across datasets
2. Resolve conflicts in overlapping data
3. Preserve unique information from all sources
4. Maintain referential integrity

**INPUT VARIABLES**:
- data_context: {data_context}
- merge_strategy: {merge_strategy}
- key_fields: {key_fields}

**EXPECTED OUTPUT**:
Return merged dataset with conflict resolution log."""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "merge_strategy", "key_fields"],
            template_format="data_merging",
            expected_output="merged_dataset",
            model_preference="gpt-4",
            temperature=0.1,
            max_tokens=1200,
            stop_sequences=[]
        )
    
    def _create_quality_check_template(self, context: Dict[str, Any], job_id: str) -> GeneratedPromptTemplate:
        """Create template for data quality assessment"""
        
        template_content = """You are Fixxy, a data quality assessment specialist. Evaluate dataset quality across multiple dimensions.

**TASK**: Comprehensive data quality assessment

**INSTRUCTIONS**:
1. Assess completeness, accuracy, consistency, and validity
2. Generate quality scores for each dimension
3. Identify specific quality issues
4. Provide improvement recommendations

**INPUT VARIABLES**:
- data_context: {data_context}
- quality_dimensions: {quality_dimensions}

**EXPECTED OUTPUT**:
Return detailed quality assessment with scores and recommendations."""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "quality_dimensions"],
            template_format="quality_assessment",
            expected_output="quality_report",
            model_preference="gpt-4",
            temperature=0.1,
            max_tokens=1000,
            stop_sequences=[]
        )
    
    def _create_field_mapping_template(self, context: Dict[str, Any], job_id: str) -> GeneratedPromptTemplate:
        """Create template for field mapping"""
        
        template_content = """You are Fixxy, a field mapping specialist. Map fields between different data schemas while preserving semantic meaning.

**TASK**: Intelligent field mapping between schemas

**INSTRUCTIONS**:
1. Analyze source and target field schemas
2. Identify semantic matches between fields
3. Handle data type conversions
4. Map complex nested structures

**INPUT VARIABLES**:
- data_context: {data_context}
- source_schema: {source_schema}
- target_schema: {target_schema}

**EXPECTED OUTPUT**:
Return field mapping configuration with transformation rules."""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "source_schema", "target_schema"],
            template_format="schema_mapping",
            expected_output="mapping_configuration",
            model_preference="gpt-4",
            temperature=0.2,
            max_tokens=1000,
            stop_sequences=[]
        )
    
    def _create_fallback_template(self, task_type: str, context: Dict[str, Any]) -> GeneratedPromptTemplate:
        """Create fallback template for unknown tasks"""
        
        template_content = f"""You are Fixxy, a data specialist. Please analyze and process the provided data according to the task requirements.

**TASK**: {task_type}

**INSTRUCTIONS**:
1. Analyze the provided data context
2. Apply appropriate data processing techniques
3. Ensure data quality and integrity
4. Provide clear documentation of changes made

**INPUT VARIABLES**:
- data_context: {{data_context}}
- task_requirements: {{task_requirements}}

**EXPECTED OUTPUT**:
Return processed data with summary of operations performed."""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context", "task_requirements"],
            template_format="generic_processing",
            expected_output="processed_data",
            model_preference="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=800,
            stop_sequences=[]
        )
    
    def _create_error_fallback_template(self, task_type: str, context: Dict[str, Any], 
                                      error_msg: str) -> GeneratedPromptTemplate:
        """Create fallback template when errors occur"""
        
        template_content = f"""You are Fixxy, a data specialist. An error occurred during template generation, but you can still help process the data.

**TASK**: {task_type} (Error Recovery Mode)
**ERROR**: {error_msg}

**INSTRUCTIONS**:
1. Process the data using basic techniques
2. Focus on data safety and integrity
3. Provide what assistance is possible given the constraints

**INPUT VARIABLES**:
- data_context: {{data_context}}

**EXPECTED OUTPUT**:
Return best-effort data processing results."""

        return GeneratedPromptTemplate(
            template_content=template_content,
            input_variables=["data_context"],
            template_format="error_recovery",
            expected_output="processed_data",
            model_preference="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
            stop_sequences=[]
        )
    
    # Helper methods for dynamic prompt management
    async def _try_get_stored_prompt(self, task_type: str, context: Dict[str, Any], 
                                   job_id: str) -> Optional[GeneratedPromptTemplate]:
        """Try to get a stored prompt for similar context with comprehensive error handling"""
        try:
            if not self.prompt_manager:
                return None
            
            # Generate context hash for lookup with error handling
            try:
                context_str = json.dumps(context, sort_keys=True, default=str)
                context_hash = hashlib.md5(context_str.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"[{job_id}] Error generating context hash: {e}")
                # Use fallback hash
                context_hash = hashlib.md5(str(context).encode()).hexdigest()
            
            # Get stored prompt with error handling
            try:
                stored_prompt = await self.prompt_manager.get_prompt(
                    sidekick_name="fixxy",
                    task_type=task_type,
                    context_hash=context_hash
                )
            except Exception as e:
                logger.warning(f"[{job_id}] Error retrieving stored prompt: {e}")
                return None
            
            if stored_prompt:
                # Convert stored prompt to GeneratedPromptTemplate with validation
                try:
                    metadata = stored_prompt.metadata if isinstance(stored_prompt.metadata, dict) else {}
                    
                    return GeneratedPromptTemplate(
                        template_content=str(stored_prompt.prompt_content),
                        input_variables=list(stored_prompt.input_variables) if isinstance(stored_prompt.input_variables, list) else [],
                        template_format=str(metadata.get('template_format', 'standard')),
                        expected_output=str(metadata.get('expected_output', 'text')),
                        model_preference=str(metadata.get('model_preference', 'gpt-3.5-turbo')),
                        temperature=float(metadata.get('temperature', 0.7)),
                        max_tokens=int(metadata.get('max_tokens', 1000))
                    )
                except Exception as e:
                    logger.warning(f"[{job_id}] Error converting stored prompt: {e}")
                    return None
            
            return None
            
        except Exception as e:
            logger.warning(f"[{job_id}] Error retrieving stored prompt: {e}")
            return None
    
    async def _store_successful_prompt(self, task_type: str, context: Dict[str, Any], 
                                     template: GeneratedPromptTemplate, job_id: str):
        """Store successful prompt for future reuse"""
        try:
            if not self.prompt_manager:
                return
            
            # Generate context hash
            import hashlib
            import json
            context_str = json.dumps(context, sort_keys=True, default=str)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()
            
            # Create stored prompt object  
            stored_prompt = StoredPrompt(
                sidekick_name="fixxy",
                task_type=task_type,
                context_hash=context_hash,
                prompt_content=template.template_content,
                input_variables=template.input_variables,
                metadata={
                    'template_format': template.template_format,
                    'expected_output': template.expected_output,
                    'model_preference': template.model_preference,
                    'temperature': template.temperature,
                    'max_tokens': template.max_tokens,
                    'created_by_job': job_id
                }
            )
            
            await self.prompt_manager.save_prompt(stored_prompt)
            logger.info(f"[{job_id}] Stored successful prompt for {task_type}")
            
        except Exception as e:
            logger.warning(f"[{job_id}] Error storing prompt: {e}")
    
    # Template getters for LangChain integration
    def _get_deduplicate_template(self) -> str:
        """Get base template for deduplication"""
        return """Analyze the provided data for duplicate records based on the specified criteria:

Fields to match: {field_list}
Similarity threshold: {similarity_threshold}
Data context: {data_context}

Identify and group duplicate records, keeping the best version of each."""
    
    def _get_standardize_template(self) -> str:
        """Get base template for standardization"""
        return """Standardize the provided data according to the target format:

Target format: {target_format}
Field mappings: {field_mappings}
Data context: {data_context}

Apply consistent formatting and structure across all records."""
    
    def _get_validate_template(self) -> str:
        """Get base template for validation"""
        return """Validate the provided data against quality requirements:

Validation rules: {validation_rules}
Field requirements: {field_requirements}
Data context: {data_context}

Check for completeness, accuracy, and consistency issues."""
    
    def _get_cleanse_template(self) -> str:
        """Get base template for cleansing"""
        return """Cleanse the provided data to improve quality:

Cleansing rules: {cleansing_rules}
Quality standards: {quality_standards}
Data context: {data_context}

Clean data while preserving integrity and maintaining audit trail."""
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task types"""
        return self._supported_tasks.copy()
    
    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        """Get configuration for specific task type"""
        return self._task_configs.get(task_type, {
            "model_preference": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
            "requires_llm": False
        })
    
    def is_enabled(self) -> bool:
        """Check if this Sidekick is enabled"""
        return self._enabled
    
    def get_version(self) -> str:
        """Get Sidekick version"""
        return self.version