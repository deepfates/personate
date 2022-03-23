import os, json, asyncio
from typing import AsyncGenerator, Callable, Coroutine, Dict, List, Optional, Union
from urllib.parse import quote_plus
from acrossword import Document, DocumentCollection, Ranker
from .agent import Agent, get_conversation_history
from ..utils import logger
from .emojify import get_all_emojis

class ReaderAgent(Agent):
    def __init__(self, **args):
        self.document_collection = DocumentCollection(documents=[])
        self.document_queue: List[Coroutine] = []
        self.emojis = get_all_emojis()
        super().__init__(**args)
        asyncio.create_task(self.assemble_documents())
        
    @classmethod
    def from_json(cls, filename: str) -> "Agent":
        with open(filename, "r") as f:
            data = json.load(f)
        agent = cls(**data)
        agent.ranker = Ranker()
        agent.reading_list = data["reading_list"]   
        # Make the document directory
        agent.home_dir = f"{agent.name}"
        if not os.path.exists(agent.home_dir):
            os.mkdir(agent.home_dir)
        agent.knowledge_dir = f"{agent.home_dir}/knowledge"
        if not os.path.exists(agent.knowledge_dir):
            os.mkdir(agent.knowledge_dir)
        for url in agent.reading_list:
            if os.path.exists(agent.knowledge_dir + "/" + quote_plus(url) + ".json"):
                continue
            else:
                if "http" in url or "www" in url:
                    agent.add_knowledge(url, is_url=True)
                    logger.debug(f"Using url knowledge {url}")
                else:
                    agent.add_knowledge(url, is_text=True)
                    logger.debug(f"Using text knowledge {url}")    
        agent.add_knowledge_directory(agent.knowledge_dir)
        agent.activators = [o["listens_to"] for o in data["activators"]]
        agent.examples = set()
        for example in data["examples"]:
            try:
                agent_dialogue = example.pop("agent")
                user = list(example.keys())[0]
                final_example = f"""<{user}> {example[user]}\n<{agent.name}> {agent_dialogue}"""
                # print(final_example)
                agent.add_example(final_example)
            except:
                pass
        return agent

    def add_knowledge(
        self,
        filename: str,
        pre_computed: bool = False,
        is_text: bool = False,
        is_url: bool = False,
        directory: Optional[str] = None,
    ) -> None:
        doc = None
        if pre_computed:
            doc = Document.deserialise(filename)
        if not directory:
            directory = self.home_dir + "/knowledge"
        if is_url:
            doc = Document.from_url_or_file(
                source=filename,
                embedding_model=self.ranker.default_model,
                is_url=True,
                directory_to_dump=directory,
                split_into_sentences=False,
            )
        elif is_text:
            doc = Document.from_url_or_file(
                source=filename,
                embedding_model=self.ranker.default_model,
                is_file=True,
                directory_to_dump=directory,
            )
        if doc:
            self.document_queue.append(doc)

    def add_knowledge_directory(self, directory_name: str):
        files = os.listdir(directory_name)
        for f in files:
            if f.endswith(".json"):
                self.add_knowledge(f"{directory_name}/{f}", pre_computed=True)
            elif f.endswith(".txt"):
                self.add_knowledge(f"{directory_name}/{f}", is_text=True)
            else:
                logger.warning(
                    "Unrecognised file format. Try labelling it as either a .txt if it's plaintext or a .json if it's been precomputed. pdfs don't work."
                )

    async def assemble_documents(self):
        documents: tuple[Document] = await asyncio.gather(*self.document_queue)
        self.document_queue.clear()
        self.document_collection.extend_documents(list(documents))

    async def search_knowledge(self, query: str, max_chars: int = 500) -> Optional[str]:
        if self.document_collection and len(self.document_collection.documents) > 0:
            top_results = [
                r.replace("\n", " ")
                for r in await self.document_collection.search(
                    query, top=3
                )
            ]
            as_str = "\n".join(top_results)
            if as_str:
                return as_str[:max_chars]
        return None

    async def get_emoji(self, msg: str) -> str:
        top_emoji = await self.ranker.rank(texts=tuple(self.emojis.keys()), query=msg, top_k=1, model=self.ranker.default_model)
        return self.emojis[top_emoji[0]]

    async def generate_agent_response(self, msg: str):

        conversation = "\n".join(await get_conversation_history(msg))

        if self.ranker:
            examples = await self.rerank_examples(conversation[-120:])
            facts = await self.rerank_facts(conversation[-120:])
            knowledge = await self.search_knowledge(conversation[-120:])
        else:
            examples = None
            facts = None
            knowledge = None
        
        reply = await self.dialogue_generator(
            name=self.name,
            description=self.description,
            conversation=conversation,
            is_ai=self.is_ai,
            examples=examples,
            facts=facts,
            knowledge=knowledge,
            response_type=self.response_type,
            annotation=self.annotation,
        )

        reply = await self.translate(reply)
        
        return reply