"""
File: /sidekick-engine/setup/cosmos_db_setup.py

Setup script for creating and configuring Azure CosmosDB 
for dynamic prompt storage system.
"""

import asyncio
import logging
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceExistsError
from clarity_secrets import get_secret

logger = logging.getLogger(__name__)

class CosmosDBSetup:
    """Setup and configuration for CosmosDB prompt storage"""
    
    def __init__(self):
        """Initialize with connection details from Key Vault"""
        self.connection_string = get_secret("cosmos-db-connection-string")
        self.database_name = "clarity_prompts"
        self.containers = {
            "stored_prompts": {
                "partition_key": "/sidekick_name",
                "throughput": 400,
                "indexes": [
                    {
                        "kind": "Range",
                        "dataType": "String",
                        "paths": ["/task_type", "/context_hash", "/status"]
                    },
                    {
                        "kind": "Range", 
                        "dataType": "Number",
                        "paths": ["/usage_count", "/success_rate", "/avg_response_time"]
                    },
                    {
                        "kind": "Range",
                        "dataType": "String", 
                        "paths": ["/created_at", "/last_used"]
                    }
                ]
            },
            "prompt_migrations": {
                "partition_key": "/sidekick_name",
                "throughput": 400,
                "indexes": [
                    {
                        "kind": "Range",
                        "dataType": "String",
                        "paths": ["/migration_status", "/migration_date"]
                    }
                ]
            }
        }
        
        self.client = CosmosClient.from_connection_string(self.connection_string)
    
    async def setup_database(self) -> bool:
        """
        Create database and containers for prompt storage.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Setting up CosmosDB for dynamic prompt storage...")
            
            # Create database
            self.database = self.client.create_database_if_not_exists(
                id=self.database_name,
                offer_throughput=400
            )
            logger.info(f"Database '{self.database_name}' ready")
            
            # Create containers
            for container_name, config in self.containers.items():
                await self._create_container(container_name, config)
            
            # Set up indexing policies
            await self._configure_indexing()
            
            # Create sample data for testing
            await self._create_sample_data()
            
            logger.info("CosmosDB setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up CosmosDB: {str(e)}")
            return False
    
    async def _create_container(self, container_name: str, config: dict) -> None:
        """Create a container with specified configuration"""
        try:
            container = self.database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path=config["partition_key"]),
                offer_throughput=config["throughput"]
            )
            logger.info(f"Container '{container_name}' ready")
            
        except CosmosResourceExistsError:
            logger.info(f"Container '{container_name}' already exists")
        except Exception as e:
            logger.error(f"Error creating container '{container_name}': {str(e)}")
            raise
    
    async def _configure_indexing(self) -> None:
        """Configure indexing policies for optimal query performance"""
        try:
            container = self.database.get_container_client("stored_prompts")
            
            # Custom indexing policy for prompt queries
            indexing_policy = {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [
                    {"path": "/*"}
                ],
                "excludedPaths": [
                    {"path": "/prompt_template/*"},  # Exclude large text fields
                    {"path": "/performance_metrics/*"}  # Exclude nested metrics
                ],
                "compositeIndexes": [
                    [
                        {"path": "/sidekick_name", "order": "ascending"},
                        {"path": "/task_type", "order": "ascending"},
                        {"path": "/last_used", "order": "descending"}
                    ],
                    [
                        {"path": "/sidekick_name", "order": "ascending"},
                        {"path": "/task_type", "order": "ascending"},
                        {"path": "/usage_count", "order": "descending"}
                    ],
                    [
                        {"path": "/sidekick_name", "order": "ascending"},
                        {"path": "/success_rate", "order": "descending"}
                    ]
                ]
            }
            
            logger.info("Indexing policy configured for optimal query performance")
            
        except Exception as e:
            logger.warning(f"Could not configure custom indexing policy: {str(e)}")
    
    async def _create_sample_data(self) -> None:
        """Create sample prompts for testing the dynamic system"""
        try:
            from datetime import datetime
            from ..models.dynamic_prompt_models import StoredPrompt, PromptStatus
            
            container = self.database.get_container_client("stored_prompts")
            
            sample_prompts = [
                {
                    "id": "fixxy_deduplicate_sample001",
                    "prompt_id": "fixxy_deduplicate_sample001",
                    "sidekick_name": "fixxy",
                    "task_type": "deduplicate",
                    "context_hash": "sample001",
                    "prompt_template": """You are an expert data analyst specializing in duplicate detection.

Task: Identify and remove duplicate records from the dataset.

Matching Fields: {match_fields}
Cleanup Rules: {cleanup_rules}
Data Context: {data_context}

Instructions:
1. Analyze records using specified matching fields
2. Apply fuzzy matching with {similarity_threshold} threshold
3. Keep the most complete record from each duplicate group
4. Provide summary of duplicates found and removed

Output: Cleaned dataset with duplicate removal report.""",
                    "input_variables": ["match_fields", "cleanup_rules", "data_context", "similarity_threshold"],
                    "context_signature": {
                        "field_list": "email,phone,name",
                        "cleanup_rules": "fuzzy_matching:true,similarity_threshold:0.9"
                    },
                    "performance_metrics": {"sample_data": True},
                    "created_at": datetime.utcnow().isoformat(),
                    "last_used": datetime.utcnow().isoformat(),
                    "usage_count": 5,
                    "success_rate": 0.95,
                    "avg_response_time": 2.3,
                    "version": "1.0",
                    "status": PromptStatus.ACTIVE.value,
                    "tags": ["fixxy", "deduplicate", "sample"]
                },
                {
                    "id": "fixxy_format_cleanup_sample002",
                    "prompt_id": "fixxy_format_cleanup_sample002", 
                    "sidekick_name": "fixxy",
                    "task_type": "format_cleanup",
                    "context_hash": "sample002",
                    "prompt_template": """You are an expert data standardization specialist.

Task: Standardize data formats according to specified rules.

Format Rules: {format_rules}
Target Fields: {field_list}
Data Context: {data_context}

Instructions:
1. Apply format rules to all specified fields
2. Convert dates to {date_format} format
3. Standardize phone numbers to {phone_format} format
4. Report any format conversion issues

Output: Standardized dataset with format conversion summary.""",
                    "input_variables": ["format_rules", "field_list", "data_context", "date_format", "phone_format"],
                    "context_signature": {
                        "field_list": "date,phone,currency",
                        "format_rules": "date_format:ISO-8601,phone_format:international"
                    },
                    "performance_metrics": {"sample_data": True},
                    "created_at": datetime.utcnow().isoformat(),
                    "last_used": datetime.utcnow().isoformat(),
                    "usage_count": 3,
                    "success_rate": 0.92,
                    "avg_response_time": 1.8,
                    "version": "1.0", 
                    "status": PromptStatus.ACTIVE.value,
                    "tags": ["fixxy", "format_cleanup", "sample"]
                }
            ]
            
            for prompt_data in sample_prompts:
                try:
                    container.create_item(body=prompt_data)
                    logger.info(f"Created sample prompt: {prompt_data['prompt_id']}")
                except CosmosResourceExistsError:
                    logger.info(f"Sample prompt already exists: {prompt_data['prompt_id']}")
            
            logger.info("Sample data created successfully")
            
        except Exception as e:
            logger.warning(f"Could not create sample data: {str(e)}")
    
    async def verify_setup(self) -> Dict[str, Any]:
        """
        Verify that the database setup is working correctly.
        
        Returns:
            Dict with verification results
        """
        verification_results = {
            "database_accessible": False,
            "containers_ready": [],
            "sample_data_count": 0,
            "indexing_functional": False,
            "errors": []
        }
        
        try:
            # Test database access
            database = self.client.get_database_client(self.database_name)
            verification_results["database_accessible"] = True
            
            # Test each container
            for container_name in self.containers.keys():
                try:
                    container = database.get_container_client(container_name)
                    
                    # Test basic query
                    query = "SELECT VALUE COUNT(1) FROM c"
                    result = list(container.query_items(query=query, enable_cross_partition_query=True))
                    item_count = result[0] if result else 0
                    
                    verification_results["containers_ready"].append({
                        "name": container_name,
                        "accessible": True,
                        "item_count": item_count
                    })
                    
                    if container_name == "stored_prompts":
                        verification_results["sample_data_count"] = item_count
                    
                except Exception as e:
                    verification_results["containers_ready"].append({
                        "name": container_name,
                        "accessible": False,
                        "error": str(e)
                    })
                    verification_results["errors"].append(f"Container {container_name}: {str(e)}")
            
            # Test complex query for indexing
            try:
                container = database.get_container_client("stored_prompts")
                test_query = """
                SELECT c.prompt_id, c.success_rate 
                FROM c 
                WHERE c.sidekick_name = 'fixxy' 
                AND c.task_type = 'deduplicate' 
                ORDER BY c.usage_count DESC
                """
                
                list(container.query_items(query=test_query, partition_key="fixxy"))
                verification_results["indexing_functional"] = True
                
            except Exception as e:
                verification_results["errors"].append(f"Indexing test failed: {str(e)}")
            
            logger.info("Database verification completed")
            return verification_results
            
        except Exception as e:
            verification_results["errors"].append(f"Database verification failed: {str(e)}")
            logger.error(f"Database verification failed: {str(e)}")
            return verification_results
    
    async def cleanup_test_data(self) -> bool:
        """
        Clean up test and sample data from the database.
        
        Returns:
            bool: True if successful
        """
        try:
            container = self.database.get_container_client("stored_prompts")
            
            # Query for test data
            query = """
            SELECT c.id, c.sidekick_name 
            FROM c 
            WHERE ARRAY_CONTAINS(c.tags, 'sample') 
            OR ARRAY_CONTAINS(c.tags, 'test')
            """
            
            test_items = list(container.query_items(
                query=query, 
                enable_cross_partition_query=True
            ))
            
            deleted_count = 0
            for item in test_items:
                try:
                    container.delete_item(
                        item=item['id'], 
                        partition_key=item['sidekick_name']
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete test item {item['id']}: {str(e)}")
            
            logger.info(f"Cleaned up {deleted_count} test items")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up test data: {str(e)}")
            return False
    
    async def update_throughput(self, container_name: str, new_throughput: int) -> bool:
        """
        Update throughput for a container.
        
        Args:
            container_name: Name of the container
            new_throughput: New throughput value
            
        Returns:
            bool: True if successful
        """
        try:
            container = self.database.get_container_client(container_name)
            
            # Update throughput
            container.replace_throughput(throughput=new_throughput)
            
            logger.info(f"Updated throughput for {container_name} to {new_throughput}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating throughput for {container_name}: {str(e)}")
            return False


async def main():
    """Main setup function"""
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Setting up CosmosDB for Clarity Dynamic Prompt Storage...")
    
    setup = CosmosDBSetup()
    
    # Run setup
    success = await setup.setup_database()
    if not success:
        print("❌ Setup failed!")
        return
    
    print("✅ Setup completed successfully!")
    
    # Verify setup
    print("\n🔍 Verifying database setup...")
    verification = await setup.verify_setup()
    
    print(f"Database accessible: {'✅' if verification['database_accessible'] else '❌'}")
    print(f"Sample data count: {verification['sample_data_count']}")
    print(f"Indexing functional: {'✅' if verification['indexing_functional'] else '❌'}")
    
    if verification['errors']:
        print("⚠️ Errors found:")
        for error in verification['errors']:
            print(f"  - {error}")
    else:
        print("✅ All verification checks passed!")
    
    print("\n📊 Container Status:")
    for container in verification['containers_ready']:
        status = "✅" if container['accessible'] else "❌"
        count = container.get('item_count', 'unknown')
        print(f"  {status} {container['name']}: {count} items")
    
    print("\n🎯 Next Steps:")
    print("1. Update your Azure Function App settings with:")
    print("   COSMOS_DB_CONNECTION_STRING=<your_connection_string>")
    print("2. Deploy the enhanced DynamicFixxySidekick")
    print("3. Test with sample requests")
    print("4. Monitor performance in Azure Portal")


if __name__ == "__main__":
    asyncio.run(main())