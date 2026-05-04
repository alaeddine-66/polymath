from logging import Logger
from types import TracebackType
from typing import Callable, Optional, Tuple

from inference.chat_completion import ChatCompletion, Message
from inference.finish_reason import FinishReason
from langchain_groq import ChatGroq

import os
import aiohttp

class GroqChatCompletion(ChatCompletion):
    """
    Groq implementation of ChatCompletion. You should create your own
    implementation with access to your specific LLM inference back-end
    """

    def __init__(
        self,
        logger_factory: Callable[[str], Logger],
        model_name: str,
        max_gen_tokens: int,
        temperature: float,
        gpu_id: int,
    ) -> None:
        self.__logger: Logger = logger_factory(__name__)
        self.__model_name = model_name
        self.__max_gen_tokens = max_gen_tokens
        self.__temperature = temperature
        self.__initial_temperature = temperature
        self.__gpu_id = gpu_id

        self.__client: Optional[ChatGroq] = None

    async def __aenter__(self) -> "GroqChatCompletion":
        self.__client = ChatGroq(
            model=self.__model_name,
            temperature=self.__temperature,
            max_tokens=self.__max_gen_tokens,
            api_key=os.environ["GROQ_API_KEY"],
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    def set_chat_model_name(self,model_name:str) -> None:
        self.__model_name = model_name

    async def create(
        self, conversation: list[Message]) -> Tuple[FinishReason, Optional[str]]:


        if self.__client is None:
            self.__logger.error("Client not initialized")
            return FinishReason.RETRYABLE_ERROR, None
        try:
            messages = [
                (msg.role.lower(), msg.text)
                for msg in conversation
            ]

            if hasattr(self.__client, "ainvoke"):
                response = await self.__client.ainvoke(messages)
            else:
                response = await asyncio.to_thread(
                    self.__client.invoke, messages
                )

            content = response.content

            if content is None or content.strip() == "":
                return FinishReason.RETRYABLE_ERROR, None

            return FinishReason.STOPPED, content
        
        except Exception as e:
            self.__logger.error(f"Exception in Groq call: {e}")
            return FinishReason.RETRYABLE_ERROR, None


    def set_temperature(self, temperature: float) -> None:
        self.__temperature = temperature

    def reset_temperature(self) -> None:
        self.__temperature = self.__initial_temperature
