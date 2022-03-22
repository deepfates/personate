# optional: uvloop makes stuff go around 3-4 times faster
#import uvloop
#uvloop.install()
import json
import asyncio
from typing import Callable, Dict, Optional, Union, List, Set
import inspect
from . import dialogue_generator

async def get_conversation_history(
    message: str, maximum_chars: int = 800
) -> list[str]:
    # Set maximum_chars higher if you are an exorbitant glutton with tokens to spare
    messages = []
    messages.append(message)
    messages.reverse()
    return messages

class Agent:
    def __init__(
        self,
        name: str,
        avatar: str,
        introduction: str,
        is_ai: bool = False,
        **kwargs
    ):
        self.name = name
        self.description = introduction
        self.avatar_url = avatar
        self.is_ai = is_ai
        self.facts = set()
        self.response_type = None
        self.annotation = None
        self.examples = set()
        self.messages_cache: dict = {}
        self.post_translators: List[Callable] = []
        self.ranker = None
        self.dialogue_generator = dialogue_generator

    @classmethod
    def from_json(cls, filename: str) -> "Agent":
        with open(filename, "r") as f:
            data = json.load(f)
        agent = cls(**data)
        agent.activators = [o["listens_to"] for o in data["activators"]]
        agent.examples = set()
        for example in data["examples"]:
            try:
                agent_dialogue = example.pop("agent")
                user = list(example.keys())[0]
                final_example = f"""<{user}> {example[user]}\n<{agent.name}> {agent_dialogue}"""
                print(final_example)
                agent.add_example(final_example)
            except:
                pass
        return agent

    def set_response_type(self, response_type: str):
        self.response_type = response_type

    def set_annotation(self, annotation: str):
        self.annotation = annotation

    def add_examples(self, examples: list):
        self.examples.update(set(examples))

    def add_ranker(self, ranker):
        self.ranker = ranker

    def add_example(self, example: str):
        self.examples.add(example)

    def add_facts(self, facts: list):
        self.facts.update(set(facts))

    def add_fact(self, fact: str):
        self.facts.add(fact)

    def add_post_translator(self, translator: Callable):
        self.post_translators.append(translator)

    async def translate(self, response: str):
        original_response = response
        for translator in self.post_translators:
            if inspect.iscoroutinefunction(translator):
                response = await translator(response)
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, translator, response)
        self.messages_cache[response] = original_response
        return response

    def facts_as_str(self, facts) -> Optional[str]:
        if not facts:
            return None
        return "- " + "\n- ".join(facts)

    async def rerank(self, query: str, docs: tuple, max_chars: int = 180):
        try:
            _top_results = await self.ranker.rank(texts=docs, query=query, top_k=len(docs), model=self.ranker.default_model)
        except Exception as e:
            print(e)
            return []
        top_results = []
        for res in _top_results:
            top_results.append(res)
            if len(''.join(top_results)) > max_chars:
                break
        return top_results

    async def rerank_examples(self, query: str, max_chars: int = 710) -> list[str]:
        if not self.ranker:
            return []
        top_results = await self.rerank(self, query, tuple(self.examples), max_chars)
        return top_results

    async def rerank_facts(self, query: str, max_chars: int = 180) -> Optional[str]:
        if not self.ranker:
            return ""
        top_results = await self.rerank(self, query, tuple(self.facts), max_chars)
        if len(top_results) == 0:
            return ""
        return self.facts_as_str(top_results)

    async def generate_agent_response(self, msg: str):

        conversation = "\n".join(await get_conversation_history(msg))

        if self.ranker:
            examples = await self.rerank_examples(conversation[-120:])
            facts = await self.rerank_facts(conversation[-120:])
        else:
            examples = None
            facts = None

        reply = await self.dialogue_generator(
            name=self.name,
            description=self.description,
            conversation=conversation,
            is_ai=self.is_ai,
            examples=examples,
            facts=facts,
            response_type=self.response_type,
            annotation=self.annotation,
        )

        reply = await self.translate(reply)
        
        return reply

    def __repr__(self):
        return f"Agent({self.name})"


