"""
This file is a modified version for ChatGLM3-6B the original glm3_agent.py file from the langchain repo.
"""
from __future__ import annotations
from langchain.agents.agent import AgentExecutor
import json
import logging
from typing import Any, List, Sequence, Tuple, Optional, Union
from pydantic.schema import model_schema
import os
from typing import List
from configs import Agent_MODEL, MODEL_ROOT_PATH
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType, AgentExecutor
import json
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional
from server.agent.utils import tool_config_from_file
from server.agent.Tool.Weather import Weather
from server.agent.Tool.Calculator import Calculator
from server.agent.Tool.classify import Classifier
from server.agent.Tool.Decision import Decision
from server.agent.Tool.Decision import StackingAveragedModels
class ChatGLM3(LLM):
    max_token: int = 8192
    do_sample: bool = False
    temperature: float = 0.8
    top_p = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    tool_names: List = []
    has_search: bool = False

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"

    def load_model(self, model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True, device_map="auto").eval()

    def _tool_history(self, prompt: str):
        ans = []
        tool_prompts = prompt.split(
            "You have access to the following tools:\n\n")[1].split("\n\nUse a json blob")[0].split("\n")

        tool_names = [tool.split(":")[0] for tool in tool_prompts]
        self.tool_names = tool_names
        tools_json = []
        for i, tool in enumerate(tool_names):
            tool_config = tool_config_from_file(tool)
            if tool_config:
                tools_json.append(tool_config)
            else:
                ValueError(
                    f"Tool {tool} config not found! It's description is {tool_prompts[i]}"
                )

        ans.append({
            "role": "system",
            "content": "Answer the following questions as best as you can. You have access to the following tools:",
            "tools": tools_json
        })
        query = f"""{prompt.split("Human: ")[-1].strip()}"""
        return ans, query

    def _extract_observation(self, prompt: str):
        return_json = prompt.split("Observation: ")[-1].split("\nThought:")[0]
        self.history.append({
            "role": "observation",
            "content": return_json
        })
        return

    def _extract_tool(self):
        if len(self.history[-1]["metadata"]) > 0:
            metadata = self.history[-1]["metadata"]
            content = self.history[-1]["content"]
            if "tool_call" in content:
                for tool in self.tool_names:
                    if tool in metadata:
                        input_para = content.split("='")[-1].split("'")[0]
                        action_json = {
                            "action": tool,
                            "action_input": input_para
                        }
                        self.has_search = True
                        return f"""
Action: 
```
{json.dumps(action_json, ensure_ascii=False)}
```"""
        final_answer_json = {
            "action": "Final Answer",
            "action_input": self.history[-1]["content"]
        }
        self.has_search = False
        return f"""
Action: 
```
{json.dumps(final_answer_json, ensure_ascii=False)}
```"""

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        if not self.has_search:
            self.history, query = self._tool_history(prompt)
        else:
            self._extract_observation(prompt)
            query = ""
        _, self.history = self.model.chat(
            self.tokenizer,
            query,
            history=self.history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        response = self._extract_tool()
        history.append((prompt, response))
        return response

tools = [Decision()]
def initialize_agent_glm3(llm,tools=tools) -> AgentExecutor:
    loaded_tolls = []
    for tool in tools:
        if isinstance(tool, str):
            loaded_tolls.append(load_tools([tool], llm=llm)[0])
        else:
            loaded_tolls.append(tool)
    return initialize_agent(
        loaded_tolls, llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )