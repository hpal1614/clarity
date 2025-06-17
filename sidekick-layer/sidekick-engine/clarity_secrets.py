"""
Clarity Secrets Management
Handles Azure Key Vault and environment variable access
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Azure Key Vault (available in Azure runtime)
try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential
    AZURE_KEYVAULT_AVAILABLE = True
except ImportError:
    AZURE_KEYVAULT_AVAILABLE = False
    logger.warning("Azure Key Vault SDK not available - using environment variables only")

class ClaritySecrets:
    """Manages secrets from Azure Key Vault and environment variables"""
    
    def __init__(self):
        self.key_vault_client = None
        self._initialize_key_vault()
    
    def _initialize_key_vault(self):
        """Initialize Azure Key Vault client if available"""
        if not AZURE_KEYVAULT_AVAILABLE:
            return
        
        try:
            key_vault_url = os.getenv("KEY_VAULT_URL")
            if key_vault_url:
                credential = DefaultAzureCredential()
                self.key_vault_client = SecretClient(
                    vault_url=key_vault_url,
                    credential=credential
                )
                logger.info("Azure Key Vault client initialized successfully")
            else:
                logger.warning("KEY_VAULT_URL not set - using environment variables only")
        except Exception as e:
            logger.error(f"Failed to initialize Key Vault client: {e}")
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Get secret from Azure Key Vault or environment variables
        
        Args:
            secret_name: Name of the secret to retrieve
            
        Returns:
            Secret value or None if not found
        """
        # First try environment variables (for development)
        env_value = os.getenv(secret_name.upper().replace("-", "_"))
        if env_value:
            return env_value
        
        # Then try Key Vault (for production)
        if self.key_vault_client:
            try:
                secret = self.key_vault_client.get_secret(secret_name)
                return secret.value
            except Exception as e:
                logger.warning(f"Failed to get secret '{secret_name}' from Key Vault: {e}")
        
        # Return default values for known secrets (UPDATE THESE WITH YOUR REAL VALUES!)
        defaults = {
            "cosmos-db-connection-string": "COSMOS_CONNECTION_NOT_CONFIGURED",  # YOU MUST UPDATE THIS
            "sidekick-layer--engine-token": "default-engine-token-CHANGE-ME",  # YOU MUST UPDATE THIS
            "sidekick-layer--relayer-gateway-token": "default-relayer-token-CHANGE-ME"  # YOU MUST UPDATE THIS
        }
        
        default_value = defaults.get(secret_name)
        if default_value:
            logger.warning(f"Using default value for secret '{secret_name}' - UPDATE THIS IN PRODUCTION!")
            return default_value
        
        logger.error(f"Secret '{secret_name}' not found in environment or Key Vault")
        return None

# Global secrets instance
_secrets_instance = ClaritySecrets()

def get_secret(secret_name: str) -> Optional[str]:
    """
    Get a secret value
    
    Args:
        secret_name: Name of the secret
        
    Returns:
        Secret value or None
    """
    return _secrets_instance.get_secret(secret_name)

# IMPORTANT: YOU NEED TO UPDATE THESE VALUES
# 1. COSMOS_DB_CONNECTION_STRING - Your CosmosDB connection string
# 2. SIDEKICK_LAYER__ENGINE_TOKEN - Your authentication token
# 3. SIDEKICK_LAYER__RELAYER_GATEWAY_TOKEN - Your relayer token
# 4. KEY_VAULT_URL - Your Azure Key Vault URL (optional)