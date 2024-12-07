import os
from collections.abc import Generator

import pytest

from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk, LLMResultChunkDelta
from core.model_runtime.entities.message_entities import AssistantPromptMessage, SystemPromptMessage, UserPromptMessage
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.kun_lun.llm.llm import KunLunLargeLanguageModel
from tests.integration_tests.model_runtime.__mock.kun_lun import setup_kun_lun_mock


@pytest.mark.parametrize("setup_kun_lun_mock", [["none"]], indirect=True)
def test_validate_credentials(setup_kun_lun_mock):
    model = KunLunLargeLanguageModel()

    with pytest.raises(CredentialsValidateFailedError):
        model.validate_credentials(model="claude-instant-1.2", credentials={"kun_lun_api_key": "invalid_key"})

    model.validate_credentials(
        model="claude-instant-1.2", credentials={"kun_lun_api_key": os.environ.get("KUN_LUN_API_KEY")}
    )


@pytest.mark.parametrize("setup_kun_lun_mock", [["none"]], indirect=True)
def test_invoke_model(setup_kun_lun_mock):
    model = KunLunLargeLanguageModel()

    response = model.invoke(
        model="claude-instant-1.2",
        credentials={
            "kun_lun_api_key": os.environ.get("KUN_LUN_API_KEY"),
            "kun_lun_api_url": os.environ.get("KUN_LUN_API_URL"),
        },
        prompt_messages=[
            SystemPromptMessage(
                content="You are a helpful AI assistant.",
            ),
            UserPromptMessage(content="Hello World!"),
        ],
        model_parameters={"temperature": 0.0, "top_p": 1.0, "max_tokens": 10},
        stop=["How"],
        stream=False,
        user="abc-123",
    )

    assert isinstance(response, LLMResult)
    assert len(response.message.content) > 0


@pytest.mark.parametrize("setup_kun_lun_mock", [["none"]], indirect=True)
def test_invoke_stream_model(setup_kun_lun_mock):
    model = KunLunLargeLanguageModel()

    response = model.invoke(
        model="claude-instant-1.2",
        credentials={"kun_lun_api_key": os.environ.get("KUN_LUN_API_KEY")},
        prompt_messages=[
            SystemPromptMessage(
                content="You are a helpful AI assistant.",
            ),
            UserPromptMessage(content="Hello World!"),
        ],
        model_parameters={"temperature": 0.0, "max_tokens": 100},
        stream=True,
        user="abc-123",
    )

    assert isinstance(response, Generator)

    for chunk in response:
        assert isinstance(chunk, LLMResultChunk)
        assert isinstance(chunk.delta, LLMResultChunkDelta)
        assert isinstance(chunk.delta.message, AssistantPromptMessage)
        assert len(chunk.delta.message.content) > 0 if chunk.delta.finish_reason is None else True


def test_get_num_tokens():
    model = KunLunLargeLanguageModel()

    num_tokens = model.get_num_tokens(
        model="claude-instant-1.2",
        credentials={"kun_lun_api_key": os.environ.get("KUN_LUN_API_KEY")},
        prompt_messages=[
            SystemPromptMessage(
                content="You are a helpful AI assistant.",
            ),
            UserPromptMessage(content="Hello World!"),
        ],
    )

    assert num_tokens == 18
