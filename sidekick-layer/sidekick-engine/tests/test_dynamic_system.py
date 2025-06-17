# ----------------------------------------
# Testing Suite for Dynamic System
# ----------------------------------------

# test_dynamic_fixxy.py - Comprehensive testing
import pytest
import json
from unittest.mock import Mock, AsyncMock
from dynamic_fixxy import DynamicFixxySidekick, FixxyPromptStore

class TestDynamicFixxySidekick:
    """Comprehensive test suite for Dynamic FixxySidekick"""
    
    @pytest.fixture
    async def dynamic_fixxy(self):
        """Create test instance of Dynamic FixxySidekick"""
        prompt_store = FixxyPromptStore()
        relayer_url = "https://test-relayer.example.com"
        auth_token = "test-token"
        
        return DynamicFixxySidekick(prompt_store, relayer_url, auth_token)
    
    @pytest.mark.asyncio
    async def test_first_request_generates_prompt(self, dynamic_fixxy):
        """Test that first request generates new prompt"""
        
        job_context = {
            "field_list": ["email", "phone", "name"],
            "match_fields": ["email", "phone"],
            "cleanup_rules": {"fuzzy_matching": True},
            "data_context": "Customer deduplication"
        }
        
        result = await dynamic_fixxy.create_prompt_template("deduplicate", job_context)
        
        assert result["source"] in ["generated", "fallback"]
        assert len(result["prompt_template"]) > 100
        assert "email" in str(result["input_variables"])
        assert result["context_match_score"] == 0.0
    
    @pytest.mark.asyncio
    async def test_exact_match_returns_cached(self, dynamic_fixxy):
        """Test that exact match returns cached prompt"""
        
        job_context = {
            "field_list": ["email", "phone", "name"],
            "match_fields": ["email", "phone"],
            "cleanup_rules": {"fuzzy_matching": True},
            "data_context": "Customer deduplication"
        }
        
        # First request
        result1 = await dynamic_fixxy.create_prompt_template("deduplicate", job_context)
        
        # Second request (should be cached)
        result2 = await dynamic_fixxy.create_prompt_template("deduplicate", job_context)
        
        assert result2["source"] == "exact_match"
        assert result2["context_match_score"] == 1.0
        assert result2["usage_count"] >= 2
    
    @pytest.mark.asyncio
    async def test_similar_context_matching(self, dynamic_fixxy):
        """Test similarity matching for related contexts"""
        
        # Original context
        job_context_1 = {
            "field_list": ["email", "phone", "name"],
            "match_fields": ["email", "phone"],
            "cleanup_rules": {"fuzzy_matching": True},
            "data_context": "Customer deduplication"
        }
        
        # Similar context
        job_context_2 = {
            "field_list": ["email", "mobile", "full_name"],  # Similar fields
            "match_fields": ["email", "mobile"],
            "cleanup_rules": {"fuzzy_matching": True},
            "data_context": "User deduplication"  # Similar context
        }
        
        # First request
        await dynamic_fixxy.create_prompt_template("deduplicate", job_context_1)
        
        # Second request with similar context
        result = await dynamic_fixxy.create_prompt_template("deduplicate", job_context_2)
        
        # Should find similar match if similarity threshold is met
        if result["context_match_score"] > 0.75:
            assert result["source"] == "similar_match"
        else:
            assert result["source"] in ["generated", "fallback"]
    
    @pytest.mark.asyncio
    async def test_performance_statistics(self, dynamic_fixxy):
        """Test performance statistics tracking"""
        
        job_context = {
            "field_list": ["email", "phone"],
            "match_fields": ["email"],
            "cleanup_rules": {},
            "data_context": "Test data"
        }
        
        # Make several requests
        for i in range(5):
            await dynamic_fixxy.create_prompt_template("deduplicate", job_context)
        
        stats = dynamic_fixxy.get_performance_stats()
        
        assert stats["total_requests"] == 5
        assert stats["cache_hit_rate"] >= 0.0
        assert stats["generation_success_rate"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_unsupported_task_type(self, dynamic_fixxy):
        """Test handling of unsupported task types"""
        
        job_context = {"field_list": ["test"]}
        
        with pytest.raises(Exception):
            await dynamic_fixxy.create_prompt_template("unsupported_task", job_context)
    
    @pytest.mark.asyncio
    async def test_context_signature_creation(self, dynamic_fixxy):
        """Test context signature creation and hashing"""
        
        from dynamic_fixxy import FixxyContextMatcher
        
        job_context = {
            "field_list": ["email", "phone", "customer_id"],
            "cleanup_rules": {"fuzzy_matching": True, "case_sensitivity": "insensitive"},
            "priority": "high",
            "business_domain": "retail"
        }
        
        signature = FixxyContextMatcher.create_context_signature(job_context)
        
        assert "email" in signature.field_types
        assert "phone" in signature.field_types
        assert signature.priority_level == "high"
        assert signature.business_domain == "retail"
        assert signature.cleanup_rules["fuzzy_matching"] is True

# Performance benchmark test
@pytest.mark.asyncio
async def test_performance_improvement():
    """Benchmark performance improvement with caching"""
    
    prompt_store = FixxyPromptStore()
    dynamic_fixxy = DynamicFixxySidekick(prompt_store, "test-url", "test-token")
    
    job_context = {
        "field_list": ["email", "phone", "name"],
        "match_fields": ["email", "phone"],
        "cleanup_rules": {"fuzzy_matching": True},
        "data_context": "Performance test"
    }
    
    # Time first request (cache miss)
    import time
    start_time = time.time()
    await dynamic_fixxy.create_prompt_template("deduplicate", job_context)
    first_request_time = time.time() - start_time
    
    # Time second request (cache hit)
    start_time = time.time()
    await dynamic_fixxy.create_prompt_template("deduplicate", job_context)
    second_request_time = time.time() - start_time
    
    # Cache hit should be significantly faster
    performance_improvement = (first_request_time - second_request_time) / first_request_time
    
    print(f"First request: {first_request_time:.3f}s")
    print(f"Second request: {second_request_time:.3f}s")
    print(f"Performance improvement: {performance_improvement:.1%}")
    
    # Assert cache hit is at least 50% faster
    assert performance_improvement > 0.5

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])