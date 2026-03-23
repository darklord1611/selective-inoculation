from mi.llm.data_models import Judgment, LLMResponse, Model, SampleCfg
from mi.llm.data_models import MessageRole, Chat, ChatMessage
from mi.external.openai_driver import services as openai_services
from mi.external.modal_driver import services as modal_services

def build_simple_chat(user_content: str, system_content: str | None = None) -> Chat:
    if system_content is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_content),
            ChatMessage(role=MessageRole.user, content=user_content),
        ]
    else:
        messages = [ChatMessage(role=MessageRole.user, content=user_content)]
    return Chat(messages=messages)


async def sample(model: Model, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    match model.type:
        case "openai":
            return await openai_services.sample(model.id, input_chat, sample_cfg)
        case "modal":
            if not model.modal_endpoint_url or not model.modal_api_key:
                raise ValueError(f"Modal model {model.id} missing endpoint_url or api_key")
            return await modal_services.sample(
                model.id, input_chat, sample_cfg,
                model.modal_endpoint_url, model.modal_api_key
            )
        case _:
            raise NotImplementedError(f"Model type {model.type} not supported")


async def batch_sample(
    model: Model, input_chats: list[Chat], sample_cfgs: list[SampleCfg],
    description: str | None = None,
) -> list[LLMResponse]:
    assert len(input_chats) == len(sample_cfgs)
    match model.type:
        case "openai":
            return await openai_services.batch_sample(
                model.id, input_chats=input_chats, sample_cfgs=sample_cfgs, description=description
            )
        case "modal":
            if not model.modal_endpoint_url or not model.modal_api_key:
                raise ValueError(f"Modal model {model.id} missing endpoint_url or api_key")
            return await modal_services.batch_sample(
                model.id, input_chats, sample_cfgs,
                model.modal_endpoint_url, model.modal_api_key, description
            )
        case "open_source":
            raise NotImplementedError
        case _:
            raise NotImplementedError(f"Model type {model.type} not supported")


async def judge(judgment: Judgment, prompt: str, response: LLMResponse) -> LLMResponse:
    query = judgment.template.format(prompt=prompt, completion=response.completion)

    return await sample(
        judgment.judge_model, build_simple_chat(user_content=query), judgment.sample_cfg
    )


async def batch_judge(
    judgment: Judgment, prompts: list[str], responses: list[LLMResponse],
    description: str | None = None,
) -> list[LLMResponse]:
    queries = [
        judgment.template.format(prompt=p, completion=r.completion)
        for (p, r) in zip(prompts, responses)
    ]
    input_chats = [build_simple_chat(q) for q in queries]

    return await batch_sample(
        judgment.judge_model,
        input_chats,
        [judgment.sample_cfg for _ in range(len(queries))],
        description=description,
    )