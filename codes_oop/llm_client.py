import openai
import asyncio
from typing import List, Dict

class LLMClient:
    """
    A class to handle interactions with a Language Model API (like OpenAI).
    """

    def __init__(self, api_key: str):
        """
        Initialize the LLMClient with an API key.

        Args:
            api_key (str): The API key for accessing the LLM service.
        """
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def call_openai_with_retry(self, messages: List[Dict[str, str]], model: str, generation_config: Dict) -> openai.ChatCompletion: # Corrected Definition - Removed 'client' argument
        """
        Handles API retries with exponential backoff for OpenAI API calls.
        """
        retry_delay = 0.1  # Initial delay in seconds
        while True:
            try:
                # Attempt to make the API call
                generation = await self.client.chat.completions.create( # Using self.client here
                    model=model,
                    messages=messages,
                    **generation_config
                )
                return generation

            except Exception as e:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 10)  # Exponential backoff up to 10s

    async def get_prediction(self, model: str, messages: List[Dict[str, str]], generation_config: Dict) -> str:
        """
        Asynchronously fetch predictions from the LLM API.

        Args:
            model (str): The name of the language model to use.
            messages (List[Dict[str, str]]): The list of messages for the chat completion.
            generation_config (Dict): Configuration parameters for text generation.

        Returns:
            str: The predicted text content, or an empty string in case of error.
        """
        try:
            generation = await self.call_openai_with_retry(messages, model, generation_config)
            return generation.choices[0].message.content if generation else ""
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def construct_input_prompt(self, question: str) -> List[Dict[str, str]]:
        """
        Construct the input prompt for the language model.

        Args:
            question (str): The question or prompt text.

        Returns:
            List[Dict[str, str]]: A list containing a single message dictionary formatted for chat completion.
        """
        prompt = [{"role": "user", "content": question}]
        return prompt