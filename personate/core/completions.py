from pyai21 import get
from pyai21.interpret import interpret

async def default_generator_api(prompt: str) -> str:
    """This function returns the text of a prompt according to settings specialised for usage with Agents.
    :param prompt: The prompt to get the text of.
    :return: The text of the prompt."""
    res = await get(
        prompt=prompt,
        stops=[">:", "From Discord", "From IRC", "\n(", "(", "> :", ">", "<", "(Sources"],
        max=250,
        presence_penalty=0.23,
        temp=0.865,
    )
    if isinstance(res, list):
        return res[0]
    else:
        return res

async def custom_generator_api(prompt: str, maximum_similarity=70, max=400, stops=[">:", "From Discord", "From IRC", "\n(", "(", "> :", ">", "<|", "(Sources", "q:", "<0x", "<" ], presence_penalty=0.23, temp=0.865, size='j1-large') -> str:
    """This function returns the text of a prompt according to settings specialised for usage with Agents.
    :param prompt: The prompt to get the text of.
    :return: The text of the prompt."""
    @interpret(maximum_similarity=maximum_similarity, max=max, stops=stops, presence_penalty=presence_penalty, temp=temp, size=size)
    async def generate_dialogue(prompt: str) -> str:
        return prompt
    
    return await generate_dialogue(prompt)
