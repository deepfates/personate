import copy
import asyncio
from typing import Callable, List, Sequence, Union
from personate.utils.logger import logger

class Frame:
    def __init__(self, fields: List[Sequence[str]], generator_api: Callable):
        self.fields = fields
        self.field_values: dict[str, Union[str, List[str]]] = {}
        # A list of field-names and their default values if unspecified.
        # self.template = template
        # A string containing the template to be used for this frame.
        self.filters = []
        # A list of filters to be applied to the outputs.
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
