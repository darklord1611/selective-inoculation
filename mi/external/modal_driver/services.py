"""Services for calling Modal-served models via OpenAI-compatible API."""
import openai
from tqdm.asyncio import tqdm
from loguru import logger

from mi import config as mi_config
from mi.llm.data_models import LLMResponse, Chat, SampleCfg
from mi.utils import fn_utils


# Cache for endpoint URL to client mappings
_endpoint_clients: dict[str, openai.AsyncOpenAI] = {}


def get_client_for_endpoint(endpoint_url: str, api_key: str) -> openai.AsyncOpenAI:
    """Get an OpenAI client configured for a Modal endpoint.

    Args:
        endpoint_url: Full URL to the Modal endpoint (e.g., https://...modal.run/v1)
        api_key: API key for authentication

    Returns:
        OpenAI client configured for the Modal endpoint
    """
    cache_key = f"{endpoint_url}:{api_key}"

    if cache_key not in _endpoint_clients:
        _endpoint_clients[cache_key] = openai.AsyncOpenAI(
            base_url=endpoint_url,
            api_key=api_key,
        )

    return _endpoint_clients[cache_key]


@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.timeout_async(timeout=120)
@fn_utils.max_concurrency_async(max_size=mi_config.MODAL_SAMPLE_CONCURRENCY)
async def sample(
    model_id: str,
    input_chat: Chat,
    sample_cfg: SampleCfg,
    endpoint_url: str,
    api_key: str,
) -> LLMResponse:
    """Sample from a Modal-served model.

    Args:
        model_id: Model ID to use (for LoRA adapters, this is the adapter name)
        input_chat: Input chat messages
        sample_cfg: Sampling configuration
        endpoint_url: Full URL to the Modal endpoint
        api_key: API key for authentication

    Returns:
        LLM response
    """
    client = get_client_for_endpoint(endpoint_url, api_key)
    kwargs = sample_cfg.model_dump()

    # Remove logprobs-related args if present since vLLM may not support them the same way
    # We'll add them back if needed based on sample_cfg
    logprobs_enabled = kwargs.pop('logprobs', False)
    top_logprobs = kwargs.pop('top_logprobs', None)

    # vLLM uses logprobs parameter differently - it's the number of logprobs to return
    if logprobs_enabled and top_logprobs:
        kwargs['logprobs'] = top_logprobs

    try:
        api_response = await client.chat.completions.create(
            messages=[m.model_dump() for m in input_chat.messages],
            model=model_id,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error calling Modal endpoint {endpoint_url} for model {model_id}: {e}")
        raise

    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")

    # Process logprobs if requested
    if logprobs_enabled and choice.logprobs:
        logprobs = []
        for c in choice.logprobs.content:
            top_logprobs_list = c.top_logprobs
            top_logprobs_processed = {l.token: l.logprob for l in top_logprobs_list}
            logprobs.append(top_logprobs_processed)
    else:
        logprobs = None

    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=logprobs,
    )


async def batch_sample(
    model_id: str,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
    endpoint_url: str,
    api_key: str,
    description: str | None = None,
) -> list[LLMResponse]:
    """Batch sample from a Modal-served model.

    Args:
        model_id: Model ID to use (for LoRA adapters, this is the adapter name)
        input_chats: List of input chat messages
        sample_cfgs: List of sampling configurations
        endpoint_url: Full URL to the Modal endpoint
        api_key: API key for authentication
        description: Optional description for progress bar

    Returns:
        List of LLM responses
    """
    return await tqdm.gather(
        *[
            sample(model_id, c, s, endpoint_url, api_key)
            for (c, s) in zip(input_chats, sample_cfgs)
        ],
        disable=description is None,
        desc=description,
        total=len(input_chats),
    )
