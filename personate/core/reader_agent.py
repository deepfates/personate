import os, json, asyncio
from typing import Coroutine, List, Optional
from types import ModuleType 
from urllib.parse import quote_plus
from acrossword import Document, DocumentCollection, Ranker
from personate.utils.logger import logger
from personate.core.agent import Agent, get_conversation_history
from personate.core.emojify import get_all_emojis
from personate.core.frame import Prompt
from personate.swarm.swarm import Swarm

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
        # Built the prompt
        agent.prompt = Prompt(agent.name)#, size='j1-large')
        agent.prompt.set_introduction(agent.description)
        if agent.is_ai:
            agent.prompt.set_is_ai(True)
        if agent.annotation:
            agent.prompt.set_pre_response_annotation(agent.annotation)
        if not agent.response_type:
            agent.response_type = "concise, interesting and conversationally-engaging"
        agent.prompt.set_response_type(agent.response_type)
        # Collect knowledge documents
        agent.ranker = Ranker()
        agent.reading_list = data["reading_list"]   
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
        # Set keywords for the agent to respond to
        agent.activators = [o["listens_to"] for o in data["activators"]]
        # Set examples of agent dialogue
        agent.examples = set()
        for example in data["examples"]:
            try:
                agent_dialogue = example.pop("agent")
                user = list(example.keys())[0]
                final_example = f"""<{user}> {example[user]}\n<{agent.name}> {agent_dialogue}"""
                agent.add_example(final_example)
            except:
                pass
        # Build agent's python abilities
        agent.swarm = Swarm()
        abilities_module = data.get("abilities_module", None)
        if abilities_module:
            agent.add_abilities_from_library(abilities_module)
        agent.abilities_dir = f"{agent.home_dir}/abilities"
        agent.add_abilities_directory(agent.abilities_dir)
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
        return ""

    def add_abilities_from_file(self, filename: str) -> None:
        logger.debug(f"Adding abilities from file {filename}")
        self.swarm.use_module(filename)

    def add_abilities_from_library(self, module: ModuleType):
        logger.debug(f"Adding abilities from module {module.__name__}")
        self.swarm.use_module(module.__name__, register_all=True)

    def add_abilities_directory(self, directory_name: str):
        files = os.listdir(directory_name)
        for f in files:
            if f.endswith(".py"):
                importname = f"{directory_name}/{f}".replace("/",".").replace(".py","")
                logger.debug(importname)
                self.add_abilities_from_file(importname)
                
    async def get_emoji(self, msg: str) -> str:
        top_emoji = await self.ranker.rank(texts=tuple(self.emojis.keys()), query=msg, top_k=1, model=self.ranker.default_model)
        return self.emojis[top_emoji[0]]

    async def generate_agent_response(self, msg: str):
        
        api_result = await self.swarm.solve(msg[-120:])
        if api_result:
            return api_result

        examples = await self.rerank_examples(msg[-120:])
        facts = await self.rerank_facts(msg[-120:])
        knowledge = await self.search_knowledge(msg[-120:])

        self.prompt.use_examples(examples)
        self.prompt.use_facts(facts)
        self.prompt.use_knowledge(knowledge)

        reply = await self.prompt.generate_reply(
            conversation=msg[-800:],
        )

        reply = await self.translate(reply)
        
        return reply