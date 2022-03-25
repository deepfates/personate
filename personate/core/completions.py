from pyai21 import get, interpret


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

# Can't use decorator in here?
async def custom_generator_api(prompt: str, maximum_similarity=70, max=400, stops=[">:", "From Discord", "From IRC", "\n(", "(", "> :", ">", "(Sources"], presence_penalty=0.23, temp=0.865) -> str:
    """This function returns the text of a prompt according to settings specialised for usage with Agents.
    :param prompt: The prompt to get the text of.
    :return: The text of the prompt."""
    @interpret(maximum_similarity=maximum_similarity, max=max, stops=stops, presence_penalty=presence_penalty, temp=temp)
    async def generate_dialogue(prompt: str) -> str:
        return prompt
    
    res = generate_dialogue(prompt)
    if isinstance(res, list):
        return res[0]
    else:
        return res
