import copy
import asyncio
from typing import Any, Callable, List, Optional, Sequence, Union
from personate.core.completions import default_generator_api, custom_generator_api
from personate.utils.logger import logger
from personate.decos.filter import Filter, DefaultFilter


class Frame:
    def __init__(self, fields: List[Sequence[str]], generator_api: Callable):
        self.fields = fields
        # A list of field-names and their default values if unspecified.
        self.field_values: dict[str, Union[str, List[str]]] = {}
        # A list of filters to be applied to the outputs.
        self.filters = []
        self.generator_api = generator_api

    async def as_string(self) -> str:
        logger.debug(self.fields)
        logger.debug(self.field_values)
        final_string = ""
        for field in self.fields:
            try:
                val = self.field_values[field[0]]
            except KeyError:
                val = field[1]
            if len(val) == 0:
                continue
            if isinstance(val, str):
                final_string += val
            elif isinstance(val, list):
                final_string += "\n".join(val)
            final_string += "\n"
        return final_string[:-1]

    def clone(self):
        new_frame = Frame(
            fields=copy.deepcopy(self.fields), generator_api=self.generator_api
        )
        new_frame.field_values = copy.deepcopy(self.field_values)
        new_frame.filters = self.filters
        return new_frame

    async def complete(self) -> str:
        prompt = await self.as_string()
        completion = None
        for i in range(5):
            completion = await self.generator_api(prompt=prompt)
            should_reject = await asyncio.gather(
                *[
                    f.validate(
                        response=completion,
                        final_prompt=prompt,
                    )
                    for f in self.filters
                ]
            )
            logger.debug(f"The Filters and their results were:")
            for f, b in zip(self.filters, should_reject):
                logger.debug(f.__class__.__name__, b)
            if not any(should_reject):
                break
        if completion:
            return completion
        else:
            raise Exception("No completion found.")


class Prompt:
    def __init__(self, name: str, **kwargs):
        self.frame = Frame(
            fields=[
                ("introduction", ""),
                ("examples", ""),
                ("facts", ""),
                ("is_ai", ""),
                ("pre_conversation_annotation", ""),
                ("response_type", ""),
                ("current_conversation", ""),
                ("pre_response_annotation", ""),
                ("conversation", ""),
                ("reading_cue", ""),
                ("api_result", ""),
                ("speech_cue", f"<{name}>:"),
            ],
            generator_api=custom_generator_api,
        )
        self.name = name
        self.frame.filters = [DefaultFilter()]
        self.max_characters: int = 1000
        self.__dict__.update(kwargs)

    def add_filter(self, filter: Filter):
        self.frame.filters.append(filter)

    def set_pre_conversation_annotation(self, annotation: str):
        self.frame.field_values["pre_conversation_annotation"] = annotation

    def set_response_type(self, response_type: str):
        self.frame.field_values[
            "response_type"
        ] = f"And now, the full 2000-word dialog where we give the character its unique, distinct voice and typing style. Users submitted questions to us and had long conversations, and we gave responses that were {response_type} (luckily we had expert researchers and specialists on the team – sometimes it took us up to three hours to craft the perfect answer):"

    def set_pre_response_annotation(self, annotation: str):
        self.frame.field_values[
            "pre_response_annotation"
        ] = f"(Quick note, and we promise there won't be any more commentary after this: {annotation})\n"

    def set_introduction(self, introduction: str):
        self.frame.field_values["introduction"] = f"Something that our team enjoyed recently was being given randomly-generated character descriptions, then writing rich, detailed, convincing dialogues. The plot-twist: those dialogues occur in a modern Discord chatroom. So, we present to you, the character description:\n\n{introduction}"

    def set_is_ai(self, is_ai: bool = False):
        ai_sentence = "Note that despite being specified as an AI, we chose to act as a human-level AI and to speak naturally, with artistic flair and personality. "
        self.frame.field_values["is_ai"] = ai_sentence

    def use_examples(self, examples: Optional[list] = None):
        if examples:
            examples_sentence = (
                "\n\nHere are some example dialogues that we sketched out that really capture the voice and tonality of the character:\n"
                + "\n".join(examples)
                + "\n\n"
            )
            self.frame.field_values["examples"] = examples_sentence
        else:
            self.frame.field_values["examples"] = ""

    def use_facts(self, facts: str):
        if facts:
            facts_sentence = f"\nWe were also given these facts, which we were told to be absolutely consistent with:\n{facts}"
            self.frame.field_values["facts"] = facts_sentence
        else:
            self.frame.facts = ""

    def use_knowledge(self, knowledge: str):
        if knowledge:
            self.frame.field_values["reading_cue"] = f'(Sources: "{knowledge}")'
        else:
            self.frame.field_values["reading_cue"] = ""
            
    def use_api_result(self, result: str):
        self.frame.field_values["api_result"] = f'(API result: "{result}")'

    async def generate_reply(self, conversation) -> str:
        self.frame.field_values["conversation"] = conversation
        completion = await self.frame.complete()
        return completion
