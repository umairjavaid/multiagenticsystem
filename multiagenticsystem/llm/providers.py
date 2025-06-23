"""
LLM provider abstraction for multiple backends.
"""

import os
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


async def retry_with_exponential_backoff(
    func,
    retry_config: RetryConfig = None,
    retryable_exceptions: tuple = None
):
    """Retry function with exponential backoff."""
    if retry_config is None:
        retry_config = RetryConfig()
    
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)
    
    for attempt in range(retry_config.max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            if attempt == retry_config.max_retries:
                logger.error(f"Max retries ({retry_config.max_retries}) exceeded: {e}")
                raise
            
            delay = min(
                retry_config.base_delay * (retry_config.exponential_base ** attempt),
                retry_config.max_delay
            )
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)


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
        usage: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.usage = usage or {}
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls or []
        
    def __str__(self) -> str:
        return self.content
        
    def __repr__(self) -> str:
        return f"LLMResponse(content='{self.content[:50]}...', usage={self.usage})"
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.config = kwargs
        self.retry_config = retry_config or RetryConfig()
        
        # Validate configuration on initialization
        if not self.validate_config():
            raise ValueError(f"Invalid configuration for {self.__class__.__name__}")
    
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
    
    async def execute_with_retry(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute with automatic retry logic."""
        async def _execute():
            return await self.execute(messages, context, **kwargs)
        
        return await retry_with_exponential_backoff(
            _execute,
            self.retry_config,
            retryable_exceptions=(ConnectionError, TimeoutError, RuntimeError)
        )
    
    def get_supported_parameters(self) -> List[str]:
        """Get list of supported parameters for this provider."""
        return ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    
    def _sanitize_parameters(self, **kwargs) -> Dict[str, Any]:
        """Sanitize parameters to only include supported ones."""
        supported = self.get_supported_parameters()
        return {k: v for k, v in kwargs.items() if k in supported}


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", **kwargs):
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        # Set attributes before calling super().__init__ for validation
        self.organization = kwargs.get("organization") or os.getenv("OPENAI_ORG_ID")
        self.base_url = kwargs.get("base_url")
        self.timeout = kwargs.get("timeout", 60)
        super().__init__(model, api_key, **kwargs)
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
    
    def get_supported_parameters(self) -> List[str]:
        """Get supported parameters for OpenAI."""
        return [
            "temperature", "max_tokens", "top_p", "frequency_penalty", 
            "presence_penalty", "stop", "logit_bias", "user", "seed"
        ]
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute OpenAI completion."""
        try:
            # Import OpenAI client
            try:
                from openai import AsyncOpenAI
                from openai import RateLimitError, APITimeoutError, APIConnectionError
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            
            # Create OpenAI client
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout
            }
            if self.organization:
                client_kwargs["organization"] = self.organization
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                
            client = AsyncOpenAI(**client_kwargs)
            
            # Sanitize and prepare parameters
            sanitized_params = self._sanitize_parameters(**kwargs)
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": sanitized_params.get("temperature", self.config.get("temperature", 0.7)),
                "max_tokens": sanitized_params.get("max_tokens", self.config.get("max_tokens", 1000)),
                "top_p": sanitized_params.get("top_p", self.config.get("top_p", 1.0)),
                "frequency_penalty": sanitized_params.get("frequency_penalty", self.config.get("frequency_penalty", 0.0)),
                "presence_penalty": sanitized_params.get("presence_penalty", self.config.get("presence_penalty", 0.0)),
            }
            
            # Add optional parameters if provided
            for param in ["stop", "logit_bias", "user", "seed"]:
                value = sanitized_params.get(param, self.config.get(param))
                if value is not None:
                    params[param] = value
            
            # Add tools if provided in context
            if context and "tools" in context:
                params["tools"] = context["tools"]
                params["tool_choice"] = context.get("tool_choice", "auto")
            
            # Make API call with specific error handling
            try:
                response = await client.chat.completions.create(**params)
            except RateLimitError as e:
                logger.warning(f"OpenAI rate limit hit: {e}")
                raise
            except APITimeoutError as e:
                logger.warning(f"OpenAI timeout: {e}")
                raise TimeoutError(f"OpenAI API timeout: {e}")
            except APIConnectionError as e:
                logger.warning(f"OpenAI connection error: {e}")
                raise ConnectionError(f"OpenAI connection failed: {e}")
            
            # Extract content
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            # Extract tool calls
            tool_calls = []
            if response.choices[0].message.tool_calls:
                tool_calls = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in response.choices[0].message.tool_calls
                ]
            
            # Build metadata
            metadata = {
                "provider": "openai",
                "model": self.model,
                "finish_reason": finish_reason,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            }
            
            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            return LLMResponse(
                content=content,
                metadata=metadata,
                usage=usage,
                finish_reason=finish_reason,
                tool_calls=tool_calls
            )
            
        except (ImportError, ValueError) as e:
            raise e  # Re-raise configuration errors
        except Exception as e:
            logger.error(f"OpenAI execution failed: {e}")
            raise RuntimeError(f"OpenAI execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        return bool(self.api_key and self.model)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model, api_key, **kwargs)
        
        self.base_url = kwargs.get("base_url")
        self.timeout = kwargs.get("timeout", 60)
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
    
    def get_supported_parameters(self) -> List[str]:
        """Get supported parameters for Anthropic."""
        return ["temperature", "max_tokens", "top_p", "top_k", "stop_sequences"]
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Anthropic completion."""
        try:
            # Import Anthropic client
            try:
                from anthropic import AsyncAnthropic
                from anthropic import RateLimitError, APITimeoutError, APIConnectionError
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
            
            # Create Anthropic client
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                
            client = AsyncAnthropic(**client_kwargs)
            
            # Convert messages format (Anthropic has different format)
            # Separate system message from conversation
            system_message = ""
            conversation_messages = []
            
            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content", "")
                else:
                    conversation_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Ensure we have at least one user message and proper alternating pattern
            if not conversation_messages:
                conversation_messages = [{"role": "user", "content": "Hello"}]
            elif conversation_messages[-1]["role"] != "user":
                conversation_messages.append({"role": "user", "content": "Please respond."})
            
            # Fix alternating pattern
            fixed_messages = []
            last_role = None
            for msg in conversation_messages:
                if msg["role"] == last_role:
                    # Merge consecutive messages from same role
                    if fixed_messages:
                        fixed_messages[-1]["content"] += f"\n\n{msg['content']}"
                    else:
                        fixed_messages.append(msg)
                else:
                    fixed_messages.append(msg)
                    last_role = msg["role"]
            
            # Sanitize and prepare parameters
            sanitized_params = self._sanitize_parameters(**kwargs)
            params = {
                "model": self.model,
                "messages": fixed_messages,
                "max_tokens": sanitized_params.get("max_tokens", self.config.get("max_tokens", 1000)),
                "temperature": sanitized_params.get("temperature", self.config.get("temperature", 0.7)),
                "top_p": sanitized_params.get("top_p", self.config.get("top_p", 1.0)),
            }
            
            # Add optional parameters
            for param in ["top_k", "stop_sequences"]:
                value = sanitized_params.get(param, self.config.get(param))
                if value is not None:
                    params[param] = value
            
            # Add system message if present
            if system_message:
                params["system"] = system_message
            
            # Add tools if provided in context
            if context and "tools" in context:
                params["tools"] = context["tools"]
                params["tool_choice"] = context.get("tool_choice", {"type": "auto"})
            
            # Make API call with specific error handling
            try:
                response = await client.messages.create(**params)
            except RateLimitError as e:
                logger.warning(f"Anthropic rate limit hit: {e}")
                raise
            except APITimeoutError as e:
                logger.warning(f"Anthropic timeout: {e}")
                raise TimeoutError(f"Anthropic API timeout: {e}")
            except APIConnectionError as e:
                logger.warning(f"Anthropic connection error: {e}")
                raise ConnectionError(f"Anthropic connection failed: {e}")
            
            # Extract content and tool calls
            content = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                        }
                    })
            
            # Build metadata
            metadata = {
                "provider": "anthropic",
                "model": self.model,
                "stop_reason": response.stop_reason,
            }
            
            # Extract usage information
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            
            return LLMResponse(
                content=content,
                metadata=metadata,
                usage=usage,
                finish_reason=response.stop_reason,
                tool_calls=tool_calls
            )
            
        except (ImportError, ValueError) as e:
            raise e  # Re-raise configuration errors
        except Exception as e:
            logger.error(f"Anthropic execution failed: {e}")
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
            # Import AWS SDK
            try:
                import boto3
                import json
            except ImportError:
                raise ImportError("AWS SDK not installed. Run: pip install boto3")
            
            # Create Bedrock client
            session = boto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            bedrock = session.client('bedrock-runtime')
            
            # Prepare payload based on model type
            if "claude" in self.model.lower():
                # Anthropic Claude format for Bedrock
                # Convert messages to Claude format
                conversation = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        conversation = f"System: {content}\n\n{conversation}"
                    elif role == "user":
                        conversation += f"Human: {content}\n\n"
                    elif role == "assistant":
                        conversation += f"Assistant: {content}\n\n"
                
                conversation += "Assistant:"
                
                payload = {
                    "prompt": conversation,
                    "max_tokens_to_sample": kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
                    "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
                    "top_p": kwargs.get("top_p", self.config.get("top_p", 1.0)),
                    "stop_sequences": kwargs.get("stop_sequences", self.config.get("stop_sequences", []))
                }
                
            elif "llama" in self.model.lower() or "mistral" in self.model.lower():
                # Meta Llama or Mistral format
                conversation = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conversation += f"<{role}>{content}</{role}>\n"
                
                payload = {
                    "prompt": conversation,
                    "max_gen_len": kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
                    "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
                    "top_p": kwargs.get("top_p", self.config.get("top_p", 1.0)),
                }
                
            elif "titan" in self.model.lower():
                # Amazon Titan format
                input_text = ""
                for msg in messages:
                    content = msg.get("content", "")
                    input_text += content + " "
                
                payload = {
                    "inputText": input_text.strip(),
                    "textGenerationConfig": {
                        "maxTokenCount": kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
                        "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
                        "topP": kwargs.get("top_p", self.config.get("top_p", 1.0)),
                        "stopSequences": kwargs.get("stop_sequences", self.config.get("stop_sequences", []))
                    }
                }
            else:
                # Generic format
                input_text = messages[-1].get("content", "") if messages else ""
                payload = {
                    "prompt": input_text,
                    "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
                    "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
                }
            
            # Make API call
            response = bedrock.invoke_model(
                modelId=self.model,
                body=json.dumps(payload),
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract content based on model type
            if "claude" in self.model.lower():
                content = response_body.get('completion', '')
                usage = {
                    "input_tokens": response_body.get('input_tokens', 0),
                    "output_tokens": response_body.get('output_tokens', 0),
                    "total_tokens": response_body.get('input_tokens', 0) + response_body.get('output_tokens', 0)
                }
            elif "titan" in self.model.lower():
                content = response_body.get('results', [{}])[0].get('outputText', '')
                usage = {
                    "input_tokens": response_body.get('inputTextTokenCount', 0),
                    "output_tokens": response_body.get('results', [{}])[0].get('tokenCount', 0),
                    "total_tokens": response_body.get('inputTextTokenCount', 0) + response_body.get('results', [{}])[0].get('tokenCount', 0)
                }
            else:
                content = response_body.get('generation', response_body.get('outputs', [''])[0])
                usage = {
                    "input_tokens": len(str(payload).split()) * 0.75,  # Rough estimate
                    "output_tokens": len(content.split()) * 0.75,
                    "total_tokens": (len(str(payload).split()) + len(content.split())) * 0.75
                }
            
            # Build metadata
            metadata = {
                "provider": "aws_bedrock",
                "model": self.model,
                "region": self.region,
                "response_metadata": response.get('ResponseMetadata', {})
            }
            
            return LLMResponse(
                content=content,
                metadata=metadata,
                usage=usage
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
            # Import Azure OpenAI client
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            
            # Create Azure OpenAI client
            client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
                "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
                "top_p": kwargs.get("top_p", self.config.get("top_p", 1.0)),
                "frequency_penalty": kwargs.get("frequency_penalty", self.config.get("frequency_penalty", 0.0)),
                "presence_penalty": kwargs.get("presence_penalty", self.config.get("presence_penalty", 0.0)),
            }
            
            # Add tools if provided in context
            if context and "tools" in context:
                params["tools"] = context["tools"]
                params["tool_choice"] = context.get("tool_choice", "auto")
            
            # Make API call
            response = await client.chat.completions.create(**params)
            
            # Extract content
            content = response.choices[0].message.content or ""
            
            # Build metadata
            metadata = {
                "provider": "azure_openai",
                "model": self.model,
                "endpoint": self.endpoint,
                "api_version": self.api_version,
                "finish_reason": response.choices[0].finish_reason,
            }
            
            # Add tool calls if present
            if response.choices[0].message.tool_calls:
                metadata["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in response.choices[0].message.tool_calls
                ]
            
            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            return LLMResponse(
                content=content,
                metadata=metadata,
                usage=usage
            )
            
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate Azure OpenAI configuration."""
        return bool(self.api_key and self.endpoint and self.model)


class TogetherProvider(LLMProvider):
    """Together AI LLM provider."""
    
    def __init__(self, model: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs):
        api_key = kwargs.get("api_key") or os.getenv("TOGETHER_API_KEY")
        super().__init__(model, api_key, **kwargs)
        
        if not self.api_key:
            raise ValueError("Together AI API key not provided")
        
        self.base_url = kwargs.get("base_url", "https://api.together.xyz/v1")
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Together AI completion."""
        try:
            # Import OpenAI client (Together AI uses OpenAI-compatible API)
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            
            # Create client with Together AI endpoint
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
                "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
                "top_p": kwargs.get("top_p", self.config.get("top_p", 1.0)),
                "frequency_penalty": kwargs.get("frequency_penalty", self.config.get("frequency_penalty", 0.0)),
                "presence_penalty": kwargs.get("presence_penalty", self.config.get("presence_penalty", 0.0)),
                "stream": False
            }
            
            # Make API call
            response = await client.chat.completions.create(**params)
            
            # Extract content
            content = response.choices[0].message.content or ""
            
            # Build metadata
            metadata = {
                "provider": "together",
                "model": self.model,
                "finish_reason": response.choices[0].finish_reason,
            }
            
            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            
            return LLMResponse(
                content=content,
                metadata=metadata,
                usage=usage
            )
            
        except Exception as e:
            raise RuntimeError(f"Together AI execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate Together AI configuration."""
        return bool(self.api_key and self.model)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models."""
    
    def __init__(self, model: str = "llama3", **kwargs):
        # Set attributes before calling super().__init__ so validate_config works
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.timeout = kwargs.get("timeout", 300)  # 5 minutes default timeout
        super().__init__(model, None, **kwargs)  # No API key needed for local Ollama
    
    def get_supported_parameters(self) -> List[str]:
        """Get supported parameters for Ollama."""
        return ["temperature", "top_p", "top_k", "num_predict", "repeat_penalty", "seed"]
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Ollama completion."""
        try:
            # Import aiohttp for HTTP requests or use fallback
            try:
                import aiohttp
                use_aiohttp = True
            except ImportError:
                logger.warning("aiohttp not available, using urllib for HTTP requests")
                import urllib.request
                import urllib.parse
                use_aiohttp = False
            
            # Convert messages to prompt format
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt = f"System: {content}\n\n{prompt}"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            
            prompt += "Assistant: "
            
            # Sanitize and prepare parameters
            sanitized_params = self._sanitize_parameters(**kwargs)
            options = {
                "temperature": sanitized_params.get("temperature", self.config.get("temperature", 0.7)),
                "top_p": sanitized_params.get("top_p", self.config.get("top_p", 1.0)),
                "num_predict": sanitized_params.get("max_tokens", self.config.get("max_tokens", 1000)),
            }
            
            # Add optional parameters
            for param in ["top_k", "repeat_penalty", "seed"]:
                value = sanitized_params.get(param, self.config.get(param))
                if value is not None:
                    options[param] = value
            
            # Prepare payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options
            }
            
            # Make API call
            if use_aiohttp:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(f"Ollama API returned status {response.status}: {error_text}")
                        
                        result = await response.json()
            else:
                # Fallback to urllib (synchronous)
                import json as json_lib
                data = json_lib.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    f"{self.base_url}/api/generate",
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Ollama API returned status {response.status}")
                    result = json_lib.loads(response.read().decode('utf-8'))
            
            # Extract content
            content = result.get("response", "")
            
            # Build metadata
            metadata = {
                "provider": "ollama",
                "model": self.model,
                "done": result.get("done", False),
                "eval_count": result.get("eval_count", 0),
                "eval_duration": result.get("eval_duration", 0),
                "load_duration": result.get("load_duration", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "prompt_eval_duration": result.get("prompt_eval_duration", 0),
            }
            
            # Extract usage information
            usage = {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
            }
            
            return LLMResponse(
                content=content,
                metadata=metadata,
                usage=usage,
                finish_reason="stop" if result.get("done", False) else "length"
            )
            
        except Exception as e:
            logger.error(f"Ollama execution failed: {e}")
            raise RuntimeError(f"Ollama execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate Ollama configuration."""
        return bool(self.model and self.base_url)


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference API provider."""
    
    def __init__(self, model: str = "microsoft/DialoGPT-medium", **kwargs):
        api_key = kwargs.get("api_key") or os.getenv("HUGGINGFACE_API_KEY")
        super().__init__(model, api_key, **kwargs)
        
        if not self.api_key:
            raise ValueError("Hugging Face API key not provided")
        
        self.base_url = kwargs.get("base_url", "https://api-inference.huggingface.co")
        self.timeout = kwargs.get("timeout", 60)
    
    def get_supported_parameters(self) -> List[str]:
        """Get supported parameters for Hugging Face."""
        return ["temperature", "max_new_tokens", "top_p", "top_k", "repetition_penalty", "seed"]
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Hugging Face completion."""
        try:
            # Import aiohttp for HTTP requests or use fallback
            try:
                import aiohttp
                use_aiohttp = True
            except ImportError:
                logger.warning("aiohttp not available, using urllib for HTTP requests")
                import urllib.request
                import urllib.parse
                use_aiohttp = False
            
            # Convert messages to text input
            input_text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    input_text = f"System: {content}\n\n{input_text}"
                elif role == "user":
                    input_text += f"User: {content}\n"
                elif role == "assistant":
                    input_text += f"Assistant: {content}\n"
            
            # Sanitize and prepare parameters
            sanitized_params = self._sanitize_parameters(**kwargs)
            parameters = {
                "temperature": sanitized_params.get("temperature", self.config.get("temperature", 0.7)),
                "max_new_tokens": sanitized_params.get("max_tokens", self.config.get("max_tokens", 1000)),
                "top_p": sanitized_params.get("top_p", self.config.get("top_p", 1.0)),
                "return_full_text": False
            }
            
            # Add optional parameters
            for param in ["top_k", "repetition_penalty", "seed"]:
                value = sanitized_params.get(param, self.config.get(param))
                if value is not None:
                    parameters[param] = value
            
            # Prepare payload
            payload = {
                "inputs": input_text.strip(),
                "parameters": parameters
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            if use_aiohttp:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/models/{self.model}",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 503:
                            error_text = await response.text()
                            raise RuntimeError(f"Model is loading. Please try again in a few minutes: {error_text}")
                        elif response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(f"Hugging Face API returned status {response.status}: {error_text}")
                        
                        result = await response.json()
            else:
                # Fallback to urllib (synchronous)
                import json as json_lib
                data = json_lib.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    f"{self.base_url}/models/{self.model}",
                    data=data,
                    headers=headers
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        if response.status != 200:
                            raise RuntimeError(f"Hugging Face API returned status {response.status}")
                        result = json_lib.loads(response.read().decode('utf-8'))
                except urllib.error.HTTPError as e:
                    if e.code == 503:
                        raise RuntimeError("Model is loading. Please try again in a few minutes.")
                    raise RuntimeError(f"Hugging Face API error: {e}")
            
            # Extract content
            content = ""
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    content = result[0].get("generated_text", "")
                else:
                    content = str(result[0])
            elif isinstance(result, dict):
                content = result.get("generated_text", str(result))
            else:
                content = str(result)
            
            # Build metadata
            metadata = {
                "provider": "huggingface",
                "model": self.model,
                "base_url": self.base_url,
            }
            
            # Extract usage information (HF doesn't provide detailed usage)
            input_tokens = len(input_text.split()) * 0.75  # Rough estimate
            output_tokens = len(content.split()) * 0.75
            usage = {
                "prompt_tokens": int(input_tokens),
                "completion_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens),
            }
            
            return LLMResponse(
                content=content,
                metadata=metadata,
                usage=usage,
                finish_reason="stop"
            )
            
        except Exception as e:
            logger.error(f"Hugging Face execution failed: {e}")
            raise RuntimeError(f"Hugging Face execution failed: {e}")
    
    def validate_config(self) -> bool:
        """Validate Hugging Face configuration."""
        return bool(self.api_key and self.model)


# Provider registry
PROVIDER_REGISTRY = {
    LLMProviderType.OPENAI: OpenAIProvider,
    LLMProviderType.ANTHROPIC: AnthropicProvider,
    LLMProviderType.AWS_BEDROCK: AWSBedrockProvider,
    LLMProviderType.AZURE_OPENAI: AzureOpenAIProvider,
    LLMProviderType.TOGETHER: TogetherProvider,
    LLMProviderType.OLLAMA: OllamaProvider,
    LLMProviderType.HUGGINGFACE: HuggingFaceProvider,
}


def get_llm_provider(
    provider: str,
    model: str,
    **kwargs
) -> LLMProvider:
    """
    Get an LLM provider instance.
    
    Args:
        provider: Provider type (openai, anthropic, aws, azure, together, ollama, huggingface)
        model: Model name
        **kwargs: Additional configuration (api_key, timeout, retry_config, etc.)
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider is not supported or configuration is invalid
    """
    try:
        provider_type = LLMProviderType(provider.lower())
    except ValueError:
        available = ", ".join([p.value for p in LLMProviderType])
        raise ValueError(f"Unsupported provider: {provider}. Available: {available}")
    
    if provider_type not in PROVIDER_REGISTRY:
        raise ValueError(f"Provider {provider_type} not implemented")
    
    provider_class = PROVIDER_REGISTRY[provider_type]
    
    try:
        return provider_class(model=model, **kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize {provider_type} provider: {e}")
        raise ValueError(f"Failed to initialize {provider_type} provider: {e}")


def list_available_providers() -> List[str]:
    """List all available LLM providers."""
    return [provider.value for provider in LLMProviderType]


def get_provider_info(provider: str) -> Dict[str, Any]:
    """
    Get information about a specific provider.
    
    Args:
        provider: Provider type
        
    Returns:
        Dict containing provider information
    """
    try:
        provider_type = LLMProviderType(provider.lower())
    except ValueError:
        raise ValueError(f"Unsupported provider: {provider}")
    
    if provider_type not in PROVIDER_REGISTRY:
        raise ValueError(f"Provider {provider_type} not implemented")
    
    provider_class = PROVIDER_REGISTRY[provider_type]
    
    # Create a temporary instance to get supported parameters
    # Note: This might fail if required config is missing, so we catch exceptions
    try:
        temp_instance = provider_class(model="test")
        supported_params = temp_instance.get_supported_parameters()
    except:
        supported_params = ["temperature", "max_tokens", "top_p"]  # Common defaults
    
    return {
        "name": provider_type.value,
        "class": provider_class.__name__,
        "description": provider_class.__doc__ or "No description available",
        "supported_parameters": supported_params,
        "requires_api_key": provider_type not in [LLMProviderType.OLLAMA],
    }


async def health_check_provider(provider: LLMProvider) -> Dict[str, Any]:
    """
    Perform a health check on a provider.
    
    Args:
        provider: LLM provider instance
        
    Returns:
        Dict containing health check results
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Validate configuration
        config_valid = provider.validate_config()
        
        # Test with a simple message
        test_messages = [{"role": "user", "content": "Hello"}]
        
        # Try to execute a simple request
        response = await provider.execute(test_messages, max_tokens=10)
        
        end_time = asyncio.get_event_loop().time()
        response_time = end_time - start_time
        
        return {
            "status": "healthy",
            "config_valid": config_valid,
            "response_time": response_time,
            "test_response_length": len(response.content),
            "total_tokens": response.total_tokens,
            "error": None
        }
        
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        response_time = end_time - start_time
        
        return {
            "status": "unhealthy",
            "config_valid": provider.validate_config(),
            "response_time": response_time,
            "test_response_length": 0,
            "total_tokens": 0,
            "error": str(e)
        }


def create_provider_from_config(config: Dict[str, Any]) -> LLMProvider:
    """
    Create a provider from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'provider', 'model', and other options
        
    Returns:
        LLM provider instance
        
    Example:
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-...",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        provider = create_provider_from_config(config)
    """
    if "provider" not in config:
        raise ValueError("Configuration must include 'provider' field")
    
    if "model" not in config:
        raise ValueError("Configuration must include 'model' field")
    
    provider_type = config.pop("provider")
    model = config.pop("model")
    
    return get_llm_provider(provider_type, model, **config)
