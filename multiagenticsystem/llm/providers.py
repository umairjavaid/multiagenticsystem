"""
LLM provider abstraction for multiple backends.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws"
    AZURE_OPENAI = "azure"
    TOGETHER = "together"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class LLMResponse:
    """Response from an LLM provider."""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.usage = usage or {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute a completion with the LLM provider."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", **kwargs):
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        super().__init__(model, api_key, **kwargs)
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute OpenAI completion."""
        try:
            # Mock implementation - replace with actual OpenAI API call
            # from openai import AsyncOpenAI
            # client = AsyncOpenAI(api_key=self.api_key)
            
            # For now, return a mock response
            content = f"OpenAI {self.model} response to: {messages[-1].get('content', '')}"
            
            return LLMResponse(
                content=content,
                metadata={"provider": "openai", "model": self.model},
                usage={"prompt_tokens": 10, "completion_tokens": 20}
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        return bool(self.api_key and self.model)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, model: str = "claude-3.5-sonnet", **kwargs):
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model, api_key, **kwargs)
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Anthropic completion."""
        try:
            # Mock implementation - replace with actual Anthropic API call
            content = f"Anthropic {self.model} response to: {messages[-1].get('content', '')}"
            
            return LLMResponse(
                content=content,
                metadata={"provider": "anthropic", "model": self.model},
                usage={"input_tokens": 10, "output_tokens": 20}
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration."""
        return bool(self.api_key and self.model)


class AWSBedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider."""
    
    def __init__(self, model: str = "anthropic.claude-3-sonnet", **kwargs):
        super().__init__(model, **kwargs)
        self.region = kwargs.get("region", "us-east-1")
        self.access_key = kwargs.get("access_key") or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = kwargs.get("secret_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute AWS Bedrock completion."""
        try:
            # Mock implementation - replace with actual Bedrock API call
            content = f"AWS Bedrock {self.model} response to: {messages[-1].get('content', '')}"
            
            return LLMResponse(
                content=content,
                metadata={"provider": "aws_bedrock", "model": self.model, "region": self.region},
                usage={"input_tokens": 10, "output_tokens": 20}
            )
            
        except Exception as e:
            raise RuntimeError(f"AWS Bedrock execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate AWS Bedrock configuration."""
        return bool(self.model and self.region)


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM provider."""
    
    def __init__(self, model: str = "gpt-35-turbo", **kwargs):
        api_key = kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        super().__init__(model, api_key, **kwargs)
        
        self.endpoint = kwargs.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = kwargs.get("api_version", "2023-12-01-preview")
        
        if not self.api_key or not self.endpoint:
            raise ValueError("Azure OpenAI API key and endpoint required")
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Azure OpenAI completion."""
        try:
            # Mock implementation - replace with actual Azure OpenAI API call
            content = f"Azure OpenAI {self.model} response to: {messages[-1].get('content', '')}"
            
            return LLMResponse(
                content=content,
                metadata={"provider": "azure_openai", "model": self.model, "endpoint": self.endpoint},
                usage={"prompt_tokens": 10, "completion_tokens": 20}
            )
            
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate Azure OpenAI configuration."""
        return bool(self.api_key and self.endpoint and self.model)


# Provider registry
PROVIDER_REGISTRY = {
    LLMProviderType.OPENAI: OpenAIProvider,
    LLMProviderType.ANTHROPIC: AnthropicProvider,
    LLMProviderType.AWS_BEDROCK: AWSBedrockProvider,
    LLMProviderType.AZURE_OPENAI: AzureOpenAIProvider,
}


def get_llm_provider(
    provider: str,
    model: str,
    **kwargs
) -> LLMProvider:
    """
    Get an LLM provider instance.
    
    Args:
        provider: Provider type (openai, anthropic, aws, azure)
        model: Model name
        **kwargs: Additional configuration
        
    Returns:
        LLMProvider instance
    """
    try:
        provider_type = LLMProviderType(provider.lower())
    except ValueError:
        raise ValueError(f"Unsupported provider: {provider}")
    
    if provider_type not in PROVIDER_REGISTRY:
        raise ValueError(f"Provider {provider_type} not implemented")
    
    provider_class = PROVIDER_REGISTRY[provider_type]
    return provider_class(model=model, **kwargs)


def list_available_providers() -> List[str]:
    """List all available LLM providers."""
    return [provider.value for provider in LLMProviderType]
