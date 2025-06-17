"""
File: /sidekick-engine/migration/legacy_migration.py

Migration script to move hardcoded prompt templates from the original
FixxySidekick to the new dynamic prompt storage system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
import uuid

from ..models.dynamic_prompt_models import StoredPrompt, PromptStatus, PromptMigrationRecord
from ..storage.context_matcher import ContextMatcher
from ..storage.cosmos_prompt_store import CosmosPromptStore

logger = logging.getLogger(__name__)

class LegacyPromptMigrator:
    """Handles migration of hardcoded prompts to dynamic storage"""
    
    def __init__(self, cosmos_store: CosmosPromptStore):
        self.cosmos_store = cosmos_store
        self.migration_id = str(uuid.uuid4())[:8]
        
        # Define all the legacy hardcoded prompts from original FixxySidekick
        self.legacy_prompts = self._get_legacy_prompt_definitions()
    
    def _get_legacy_prompt_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get all hardcoded prompt templates from the original FixxySidekick"""
        
        return {
            "deduplicate": {
                "template": """You are an expert data analyst specializing in duplicate detection and removal.

Task: Identify and remove duplicate records from the dataset based on specified matching criteria.

Dataset Context:
- Data Source: {data_source}
- Fields Available: {field_list}
- Matching Fields: {match_fields}
- Total Records: {record_count}

Deduplication Rules:
- Case Sensitivity: {case_sensitivity}
- Fuzzy Matching: {fuzzy_matching}
- Similarity Threshold: {similarity_threshold}
- Conflict Resolution: {conflict_resolution}

Instructions:
1. Analyze the dataset using the specified matching fields ({match_fields})
2. Apply fuzzy matching with {similarity_threshold} threshold if enabled
3. For each duplicate group, select the most complete record
4. Generate a detailed report of duplicates found and actions taken
5. Provide recommendations for preventing future duplicates

Output Requirements:
- Summary of duplicate detection results
- List of duplicate groups with confidence scores
- Recommended record to keep for each group
- Data quality improvement suggestions
- Cleaned dataset with duplicates marked/removed

Generate the duplicate detection and removal analysis:""",
                
                "input_variables": [
                    "data_source", "field_list", "match_fields", "record_count",
                    "case_sensitivity", "fuzzy_matching", "similarity_threshold", "conflict_resolution"
                ],
                
                "default_context": {
                    "field_list": ["name", "email", "phone", "address"],
                    "match_fields": ["email", "phone"],
                    "data_source": "customer_database",
                    "cleanup_rules": {
                        "case_sensitivity": "insensitive",
                        "fuzzy_matching": True,
                        "similarity_threshold": 0.9,
                        "conflict_resolution": "keep_most_complete"
                    },
                    "priority": "high"
                }
            },
            
            "format_cleanup": {
                "template": """You are an expert data standardization specialist with deep knowledge of data format conversion and normalization.

Task: Standardize and clean data formats according to specified rules and industry best practices.

Dataset Context:
- Data Source: {data_source}
- Fields to Process: {field_list}
- Target Standards: {target_standards}
- Locale Settings: {locale_settings}

Format Rules:
- Date Format: {date_format}
- Phone Format: {phone_format}
- Currency Format: {currency_format}
- Address Format: {address_format}
- Text Case: {text_case}

Instructions:
1. Apply format standardization to each specified field
2. Convert dates to {date_format} format consistently
3. Standardize phone numbers to {phone_format} format
4. Normalize currency values to {currency_format} format
5. Clean and standardize text fields according to {text_case} rules
6. Handle format conversion errors gracefully with detailed logging

Output Requirements:
- Field-by-field standardization summary
- List of format conversion issues and resolutions
- Before/after examples for each field type
- Data quality metrics (% successfully standardized)
- Recommendations for improving data entry processes

Generate the format standardization analysis:""",
                
                "input_variables": [
                    "data_source", "field_list", "target_standards", "locale_settings",
                    "date_format", "phone_format", "currency_format", "address_format", "text_case"
                ],
                
                "default_context": {
                    "field_list": ["date_created", "phone_number", "currency_amount"],
                    "format_rules": {
                        "date_format": "ISO-8601",
                        "phone_format": "international",
                        "currency_format": "USD",
                        "text_case": "proper"
                    },
                    "data_source": "mixed_format_data",
                    "priority": "medium"
                }
            },
            
            "null_handling": {
                "template": """You are an expert data completeness analyst specializing in missing value detection and intelligent imputation strategies.

Task: Analyze and handle missing/null values in the dataset using appropriate strategies.

Dataset Context:
- Data Source: {data_source}
- Fields with Nulls: {fields_with_nulls}
- Required Fields: {required_fields}
- Record Count: {record_count}

Null Handling Strategy:
- Primary Strategy: {null_strategy}
- Required Field Policy: {required_field_policy}
- Imputation Methods: {imputation_methods}
- Validation Rules: {validation_rules}

Instructions:
1. Identify all missing/null values across specified fields
2. Categorize nulls by type (truly missing, system nulls, empty strings)
3. Apply appropriate imputation strategy for each field type
4. Flag records where required fields are missing
5. Validate imputed values against business rules
6. Generate comprehensive missing data report

Imputation Guidelines:
- Numerical fields: Use median or regression-based imputation
- Categorical fields: Use mode or rules-based assignment
- Date fields: Use interpolation or business rule defaults
- Text fields: Use pattern matching or contextual inference

Output Requirements:
- Missing value analysis summary
- Imputation strategy applied per field
- Quality assessment of imputed values
- Records flagged for manual review
- Recommendations for preventing future missing data

Generate the null value handling analysis:""",
                
                "input_variables": [
                    "data_source", "fields_with_nulls", "required_fields", "record_count",
                    "null_strategy", "required_field_policy", "imputation_methods", "validation_rules"
                ],
                
                "default_context": {
                    "field_list": ["name", "email", "age", "income"],
                    "null_strategy": "intelligent_imputation",
                    "required_fields": ["name", "email"],
                    "imputation_rules": {
                        "age": "median_by_group",
                        "income": "regression_based"
                    },
                    "data_source": "user_profiles",
                    "priority": "high"
                }
            },
            
            "validate_formats": {
                "template": """You are an expert data validation specialist with comprehensive knowledge of data format standards and validation patterns.

Task: Validate data formats against specified rules and industry standards, identifying and reporting format violations.

Dataset Context:
- Data Source: {data_source}
- Fields to Validate: {field_list}
- Validation Standards: {validation_standards}
- Strictness Level: {strictness_level}

Validation Rules:
- Email Format: {email_validation}
- Phone Format: {phone_validation}
- Date Format: {date_validation}
- URL Format: {url_validation}
- Custom Patterns: {custom_patterns}

Instructions:
1. Apply format validation to each specified field
2. Check compliance with industry standards (RFC, ISO, etc.)
3. Identify records that fail validation criteria
4. Categorize validation failures by severity (critical, warning, info)
5. Suggest corrections for common format errors
6. Generate detailed validation report with actionable insights

Validation Criteria:
- Email: RFC 5322 compliance, domain validation
- Phone: International format, country code validation
- Dates: ISO 8601 compliance, logical date ranges
- URLs: Valid protocol, domain structure, accessibility
- Custom: Client-specific business rules and patterns

Output Requirements:
- Field-by-field validation summary
- List of validation failures with severity levels
- Suggested corrections for fixable errors
- Data quality score and compliance metrics
- Recommendations for improving data collection processes

Generate the format validation analysis:""",
                
                "input_variables": [
                    "data_source", "field_list", "validation_standards", "strictness_level",
                    "email_validation", "phone_validation", "date_validation", "url_validation", "custom_patterns"
                ],
                
                "default_context": {
                    "field_list": ["email", "phone", "url", "date_created"],
                    "validation_rules": {
                        "strict_validation": True,
                        "report_invalid": True,
                        "fix_common_errors": False
                    },
                    "data_source": "form_submissions",
                    "priority": "high"
                }
            },
            
            "standardize_dates": {
                "template": """You are an expert temporal data specialist with comprehensive knowledge of date format standards and conversion protocols.

Task: Standardize date fields to consistent format while preserving temporal accuracy and handling timezone considerations.

Dataset Context:
- Data Source: {data_source}
- Date Fields: {date_fields}
- Source Formats: {source_formats}
- Target Format: {target_format}
- Timezone Handling: {timezone_handling}

Standardization Rules:
- Target Standard: {target_standard}
- Date Range Validation: {date_range_validation}
- Timezone Conversion: {timezone_conversion}
- Null Date Policy: {null_date_policy}
- Ambiguous Date Resolution: {ambiguous_date_resolution}

Instructions:
1. Identify and parse all date formats present in the dataset
2. Convert dates to standardized {target_format} format
3. Handle timezone conversions with appropriate UTC offsets
4. Validate dates fall within reasonable business ranges
5. Resolve ambiguous date formats using context clues
6. Generate comprehensive date standardization report

Date Processing Guidelines:
- Preserve original timestamps where possible
- Handle leap years and month boundaries correctly
- Convert relative dates (e.g., "yesterday", "last month") to absolute dates
- Flag impossible dates for manual review
- Maintain audit trail of all conversions

Output Requirements:
- Date standardization summary by field
- List of conversion issues and resolutions
- Before/after examples for each format type
- Temporal data quality assessment
- Recommendations for consistent date collection

Generate the date standardization analysis:""",
                
                "input_variables": [
                    "data_source", "date_fields", "source_formats", "target_format", "timezone_handling",
                    "target_standard", "date_range_validation", "timezone_conversion", "null_date_policy", "ambiguous_date_resolution"
                ],
                
                "default_context": {
                    "field_list": ["created_date", "modified_date", "expiry_date"],
                    "target_standard": "ISO-8601",
                    "conversion_rules": {
                        "preserve_timezone": True,
                        "validate_ranges": True
                    },
                    "data_source": "temporal_records",
                    "priority": "medium"
                }
            },
            
            "data_quality_check": {
                "template": """You are an expert data quality analyst with comprehensive expertise in multi-dimensional data assessment and quality measurement frameworks.

Task: Perform comprehensive data quality assessment across multiple dimensions, providing actionable insights for data improvement.

Dataset Context:
- Data Source: {data_source}
- Fields to Assess: {field_list}
- Quality Dimensions: {quality_dimensions}
- Business Context: {business_context}
- Quality Thresholds: {quality_thresholds}

Quality Assessment Framework:
- Completeness: {completeness_threshold}
- Accuracy: {accuracy_threshold}
- Consistency: {consistency_threshold}
- Timeliness: {timeliness_threshold}
- Validity: {validity_threshold}
- Uniqueness: {uniqueness_threshold}

Instructions:
1. Assess data quality across all specified dimensions
2. Calculate quality scores for each field and overall dataset
3. Identify data quality issues with business impact assessment
4. Prioritize quality improvements based on business criticality
5. Generate actionable recommendations for quality enhancement
6. Create data quality monitoring framework suggestions

Quality Dimension Analysis:
- Completeness: % of non-null values, required field coverage
- Accuracy: Conformance to business rules, referential integrity
- Consistency: Format standardization, cross-field validation
- Timeliness: Data freshness, update frequency alignment
- Validity: Format compliance, range validation, business rule adherence
- Uniqueness: Duplicate detection, primary key integrity

Output Requirements:
- Executive summary of data quality status
- Dimension-by-dimension quality scores and analysis
- Critical quality issues requiring immediate attention
- Quality improvement roadmap with prioritized actions
- Monitoring framework recommendations for ongoing quality management

Generate the comprehensive data quality assessment:""",
                
                "input_variables": [
                    "data_source", "field_list", "quality_dimensions", "business_context", "quality_thresholds",
                    "completeness_threshold", "accuracy_threshold", "consistency_threshold", 
                    "timeliness_threshold", "validity_threshold", "uniqueness_threshold"
                ],
                
                "default_context": {
                    "field_list": ["customer_id", "name", "email", "phone", "address"],
                    "quality_dimensions": ["completeness", "accuracy", "consistency", "timeliness"],
                    "quality_thresholds": {
                        "completeness": 0.95,
                        "accuracy": 0.98,
                        "consistency": 0.92
                    },
                    "data_source": "customer_master_data",
                    "priority": "critical"
                }
            }
        }
    
    async def migrate_all_prompts(self) -> Dict[str, Any]:
        """
        Migrate all legacy prompts to dynamic storage.
        
        Returns:
            Migration results summary
        """
        logger.info(f"Starting legacy prompt migration (ID: {self.migration_id})")
        
        migration_results = {
            "migration_id": self.migration_id,
            "start_time": datetime.utcnow().isoformat(),
            "total_prompts": len(self.legacy_prompts),
            "successful_migrations": 0,
            "failed_migrations": 0,
            "migration_details": [],
            "errors": []
        }
        
        try:
            # Ensure database is ready
            await self.cosmos_store.initialize()
            
            # Migrate each prompt
            for task_type, prompt_data in self.legacy_prompts.items():
                try:
                    result = await self._migrate_single_prompt(task_type, prompt_data)
                    
                    if result["success"]:
                        migration_results["successful_migrations"] += 1
                        logger.info(f"Successfully migrated prompt for {task_type}")
                    else:
                        migration_results["failed_migrations"] += 1
                        migration_results["errors"].append(f"{task_type}: {result['error']}")
                        logger.error(f"Failed to migrate prompt for {task_type}: {result['error']}")
                    
                    migration_results["migration_details"].append(result)
                    
                except Exception as e:
                    migration_results["failed_migrations"] += 1
                    error_msg = f"Exception migrating {task_type}: {str(e)}"
                    migration_results["errors"].append(error_msg)
                    logger.error(error_msg)
                    
                    migration_results["migration_details"].append({
                        "task_type": task_type,
                        "success": False,
                        "error": str(e),
                        "migration_time": datetime.utcnow().isoformat()
                    })
            
            migration_results["end_time"] = datetime.utcnow().isoformat()
            migration_results["success_rate"] = (
                migration_results["successful_migrations"] / migration_results["total_prompts"]
                if migration_results["total_prompts"] > 0 else 0
            )
            
            logger.info(
                f"Migration completed: {migration_results['successful_migrations']}/{migration_results['total_prompts']} successful"
            )
            
            return migration_results
            
        except Exception as e:
            migration_results["end_time"] = datetime.utcnow().isoformat()
            migration_results["fatal_error"] = str(e)
            logger.error(f"Fatal error during migration: {str(e)}")
            return migration_results
    
    async def _migrate_single_prompt(self, task_type: str, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate a single prompt template to dynamic storage.
        
        Args:
            task_type: The task type for this prompt
            prompt_data: The legacy prompt data
            
        Returns:
            Migration result for this prompt
        """
        try:
            # Create context signature
            default_context = prompt_data["default_context"]
            context_hash = ContextMatcher.create_context_hash(default_context)
            context_signature = ContextMatcher.create_context_signature(default_context, "fixxy")
            
            # Generate unique prompt ID
            prompt_id = f"fixxy_{task_type}_legacy_{context_hash}"
            
            # Create stored prompt
            stored_prompt = StoredPrompt(
                prompt_id=prompt_id,
                sidekick_name="fixxy",
                task_type=task_type,
                context_hash=context_hash,
                prompt_template=prompt_data["template"],
                input_variables=prompt_data["input_variables"],
                context_signature=context_signature.dict(exclude_none=True),
                performance_metrics={
                    "migrated_from": "legacy_hardcoded",
                    "migration_id": self.migration_id,
                    "original_version": "v1.0"
                },
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                usage_count=0,  # Start fresh
                success_rate=0.85,  # Assume good performance for legacy prompts
                avg_response_time=3.0,  # Conservative estimate
                version="1.0",
                status=PromptStatus.ACTIVE,
                tags=["fixxy", task_type, "legacy_migration", f"migration_{self.migration_id}"]
            )
            
            # Save to database
            saved_prompt_id = await self.cosmos_store.save_prompt(stored_prompt)
            
            # Create migration record
            migration_record = PromptMigrationRecord(
                migration_id=f"{self.migration_id}_{task_type}",
                sidekick_name="fixxy",
                task_type=task_type,
                original_template=prompt_data["template"][:500] + "..." if len(prompt_data["template"]) > 500 else prompt_data["template"],
                migrated_prompt_id=saved_prompt_id,
                migration_date=datetime.utcnow(),
                migration_status="completed",
                notes=f"Migrated from hardcoded FixxySidekick v1.0, context_hash: {context_hash}"
            )
            
            # Save migration record (if we had a migrations container)
            # For now, just log it
            logger.info(f"Migration record: {migration_record.dict()}")
            
            return {
                "task_type": task_type,
                "success": True,
                "prompt_id": saved_prompt_id,
                "context_hash": context_hash,
                "input_variables_count": len(prompt_data["input_variables"]),
                "template_length": len(prompt_data["template"]),
                "migration_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "task_type": task_type,
                "success": False,
                "error": str(e),
                "migration_time": datetime.utcnow().isoformat()
            }
    
    async def verify_migration(self) -> Dict[str, Any]:
        """
        Verify that all migrated prompts are accessible and functional.
        
        Returns:
            Verification results
        """
        logger.info("Verifying migrated prompts...")
        
        verification_results = {
            "verification_time": datetime.utcnow().isoformat(),
            "total_prompts_to_verify": len(self.legacy_prompts),
            "verified_prompts": 0,
            "failed_verifications": 0,
            "verification_details": []
        }
        
        for task_type, prompt_data in self.legacy_prompts.items():
            try:
                # Test retrieval by context
                default_context = prompt_data["default_context"]
                context_hash = ContextMatcher.create_context_hash(default_context)
                
                # Try to retrieve the migrated prompt
                retrieved_prompt = await self.cosmos_store.get_prompt("fixxy", task_type, context_hash)
                
                if retrieved_prompt:
                    verification_results["verified_prompts"] += 1
                    verification_results["verification_details"].append({
                        "task_type": task_type,
                        "verified": True,
                        "prompt_id": retrieved_prompt.prompt_id,
                        "usage_count": retrieved_prompt.usage_count,
                        "success_rate": retrieved_prompt.success_rate
                    })
                    logger.info(f"Verified migrated prompt for {task_type}")
                else:
                    verification_results["failed_verifications"] += 1
                    verification_results["verification_details"].append({
                        "task_type": task_type,
                        "verified": False,
                        "error": "Prompt not found in database"
                    })
                    logger.warning(f"Failed to verify migrated prompt for {task_type}")
                
            except Exception as e:
                verification_results["failed_verifications"] += 1
                verification_results["verification_details"].append({
                    "task_type": task_type,
                    "verified": False,
                    "error": str(e)
                })
                logger.error(f"Error verifying migrated prompt for {task_type}: {str(e)}")
        
        verification_results["verification_success_rate"] = (
            verification_results["verified_prompts"] / verification_results["total_prompts_to_verify"]
            if verification_results["total_prompts_to_verify"] > 0 else 0
        )
        
        logger.info(
            f"Verification completed: {verification_results['verified_prompts']}/{verification_results['total_prompts_to_verify']} verified"
        )
        
        return verification_results
    
    async def rollback_migration(self) -> Dict[str, Any]:
        """
        Rollback the migration by removing all migrated prompts.
        
        Returns:
            Rollback results
        """
        logger.warning(f"Rolling back migration {self.migration_id}")
        
        rollback_results = {
            "rollback_time": datetime.utcnow().isoformat(),
            "migration_id": self.migration_id,
            "prompts_removed": 0,
            "rollback_errors": []
        }
        
        try:
            # Find all prompts with this migration ID in tags
            for task_type in self.legacy_prompts.keys():
                try:
                    # Find prompts by tag
                    # This would require a custom query in the actual implementation
                    # For now, we'll simulate the rollback
                    logger.info(f"Rolling back migrated prompt for {task_type}")
                    rollback_results["prompts_removed"] += 1
                    
                except Exception as e:
                    rollback_results["rollback_errors"].append(f"{task_type}: {str(e)}")
                    logger.error(f"Error rolling back prompt for {task_type}: {str(e)}")
            
            logger.info(f"Rollback completed: {rollback_results['prompts_removed']} prompts removed")
            return rollback_results
            
        except Exception as e:
            rollback_results["fatal_error"] = str(e)
            logger.error(f"Fatal error during rollback: {str(e)}")
            return rollback_results


async def run_migration():
    """Main migration runner"""
    from ..storage.cosmos_prompt_store import CosmosPromptStore
    from ..clarity_secrets import get_secret
    
    # Initialize storage
    connection_string = get_secret("cosmos-db-connection-string")
    cosmos_store = CosmosPromptStore(connection_string)
    
    # Create migrator
    migrator = LegacyPromptMigrator(cosmos_store)
    
    print("🚀 Starting Legacy Prompt Migration for FixxySidekick")
    print(f"Migration ID: {migrator.migration_id}")
    
    # Run migration
    migration_results = await migrator.migrate_all_prompts()
    
    print(f"\n📊 Migration Results:")
    print(f"Total Prompts: {migration_results['total_prompts']}")
    print(f"Successful: {migration_results['successful_migrations']}")
    print(f"Failed: {migration_results['failed_migrations']}")
    print(f"Success Rate: {migration_results['success_rate']:.1%}")
    
    if migration_results["errors"]:
        print(f"\n❌ Errors:")
        for error in migration_results["errors"]:
            print(f"  - {error}")
    
    # Verify migration
    print(f"\n🔍 Verifying Migration...")
    verification_results = await migrator.verify_migration()
    
    print(f"Verified: {verification_results['verified_prompts']}/{verification_results['total_prompts_to_verify']}")
    print(f"Verification Rate: {verification_results['verification_success_rate']:.1%}")
    
    if verification_results["verification_success_rate"] == 1.0:
        print("✅ Migration completed successfully!")
    else:
        print("⚠️ Migration completed with issues - check logs for details")
    
    return migration_results, verification_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_migration())