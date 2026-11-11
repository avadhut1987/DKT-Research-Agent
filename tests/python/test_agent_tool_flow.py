import unittest

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from dkt_agent.langchain_tools import search_research_evidence


class DeterministicToolCallingModel(BaseChatModel):
    calls: int = 0
    available_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "deterministic-tool-test"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.available_tools = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1

        if self.calls == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_research_evidence",
                        "args": {
                            "query": "ergodicity",
                            "max_files": 1,
                        },
                        "id": "test_call_001",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            tool_messages = [
                msg for msg in messages if isinstance(msg, ToolMessage)
            ]

            if not tool_messages:
                raise AssertionError("Model did not receive a ToolMessage.")

            message = AIMessage(
                content=(
                    "Evidence search executed successfully. "
                    + str(tool_messages[-1].content)[:1200]
                )
            )

        return ChatResult(
            generations=[ChatGeneration(message=message)]
        )


class TestAgentToolFlow(unittest.TestCase):

    def test_evidence_search_tool_execution(self):
        model = DeterministicToolCallingModel()

        agent = create_agent(
            model=model,
            tools=[search_research_evidence],
            system_prompt="Use the evidence-search tool when requested.",
        )

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": "Search the research evidence for ergodicity.",
                }
            ]
        })

        messages = result["messages"]

        tool_messages = [
            msg for msg in messages if isinstance(msg, ToolMessage)
        ]

        assistant_messages = [
            msg for msg in messages if isinstance(msg, AIMessage)
        ]

        self.assertEqual(model.calls, 2)
        self.assertIn("search_research_evidence", model.available_tools)

        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(
            tool_messages[0].name,
            "search_research_evidence",
        )

        self.assertTrue(
            any(msg.tool_calls for msg in assistant_messages),
            "The model did not request a tool call.",
        )

        self.assertTrue(
            any(
                not msg.tool_calls and msg.content
                for msg in assistant_messages
            ),
            "The model did not produce a final response.",
        )

        self.assertIn(
            '"status": "success"',
            tool_messages[0].content,
        )


if __name__ == "__main__":
    unittest.main()
