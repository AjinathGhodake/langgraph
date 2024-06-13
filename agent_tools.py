from langchain.tools import Tool

from langchain.agents import Agent, AgentOutputParser
from langchain_groq import ChatGroq
from langchain.prompts import Prompt
from langchain.chains import LLMChain
from pydantic import Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()


# Function implementations for each tool
def collect_email(query: str):
    return "Please provide your email address."


def collect_full_name(query: str):
    return "Please provide your full name."


def collect_dob(query: str):
    return "Please provide your date of birth."


def collect_address(query: str):
    return "Please provide your address."


def collect_phone_number(query: str):
    return "Please provide your phone number."


def collect_emergency_contact(query: str):
    return "Please provide your emergency contact number."


# Tool definitions
class CollectEmailTool(Tool):
    def __init__(self):
        super().__init__(
            name="collect_email",
            description="Collect the user's email address",
            func=collect_email,
        )


class CollectFullNameTool(Tool):
    def __init__(self):
        super().__init__(
            name="collect_full_name",
            description="Collect the user's full name",
            func=collect_full_name,
        )


class CollectDOBTool(Tool):
    def __init__(self):
        super().__init__(
            name="collect_dob",
            description="Collect the user's date of birth",
            func=collect_dob,
        )


class CollectAddressTool(Tool):
    def __init__(self):
        super().__init__(
            name="collect_address",
            description="Collect the user's address",
            func=collect_address,
        )


class CollectPhoneNumberTool(Tool):
    def __init__(self):
        super().__init__(
            name="collect_phone_number",
            description="Collect the user's phone number",
            func=collect_phone_number,
        )


class CollectEmergencyContactTool(Tool):
    def __init__(self):
        super().__init__(
            name="collect_emergency_contact",
            description="Collect the user's emergency contact number",
            func=collect_emergency_contact,
        )


class NoOpOutputParser(AgentOutputParser):
    def parse(self, text):
        return text


class OnboardingAgent(Agent):
    collected_data: dict = Field(default_factory=dict)
    current_tool_index: int = 0
    tools: List[Tool]

    def __init__(
        self,
        tools: List[Tool],
        llm_chain: LLMChain,
        output_parser: AgentOutputParser = None,
    ):
        super().__init__(tools=tools, llm_chain=llm_chain, output_parser=output_parser)
        self.output_parser = output_parser or NoOpOutputParser()
        self.collected_data = {}
        self.current_tool_index = 0
        self.tools = tools

    def save_user_info(self):
        # response = requests.post(
        #     "http://localhost:8000/api/users/", json=self.collected_data
        # )
        print(self.collected_data)
        if True:
            return "Your information has been saved successfully."
        else:
            return "There was an error saving your information."

    def run(self, query: str):
        if self.current_tool_index < len(self.tools):
            current_tool = self.tools[self.current_tool_index]

            # Extract user response and store in collected data
            if current_tool.name == "collect_email":
                self.collected_data["email"] = query.split(":")[-1].strip()
            elif current_tool.name == "collect_full_name":
                self.collected_data["full_name"] = query.split(":")[-1].strip()
            elif current_tool.name == "collect_dob":
                self.collected_data["date_of_birth"] = query.split(":")[-1].strip()
            elif current_tool.name == "collect_address":
                self.collected_data["address"] = query.split(":")[-1].strip()
            elif current_tool.name == "collect_phone_number":
                self.collected_data["phone_number"] = query.split(":")[-1].strip()
            elif current_tool.name == "collect_emergency_contact":
                self.collected_data["emergency_contact_number"] = query.split(":")[
                    -1
                ].strip()

            # Move to the next tool
            self.current_tool_index += 1
            if self.current_tool_index < len(self.tools):
                next_tool = self.tools[self.current_tool_index]
                return next_tool.run(query)
            else:
                # Save data if all tools have been processed
                return self.save_user_info()
        else:
            # Initial prompt
            return self.tools[0].run(query)

    def _get_default_output_parser(self):
        return NoOpOutputParser()

    def create_prompt(self):
        return Prompt(
            template="Let's start the onboarding process.", input_variables=["input"]
        )

    @property
    def llm_prefix(self):
        return "User:"

    @property
    def observation_prefix(self):
        return "Agent:"


# Define the tools
tools = [
    CollectEmailTool(),
    CollectFullNameTool(),
    CollectDOBTool(),
    CollectAddressTool(),
    CollectPhoneNumberTool(),
    CollectEmergencyContactTool(),
]


fields_to_ask = [
    "email",
    "full_name",
    "date_of_birth",
    "full_address",
    "phone_number",
]
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ.get(
        "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"
    ),
    model_name="llama3-8b-8192",
)
system_prompt_chatbot = f"""
You are Seva, an AI bot to help people not to feel lonely, and alone, you are their bestie and their companion,
people can interact with you and they should feel better and excited to reveal what they haven't in their life till now.

Initially you have to take these fields of user, {fields_to_ask}, without having all this info with you don't proceed any
conversation, this is the registration for the application,once you gather all the details from the user in the interaction,
then confirm him once that the details gathered are correct or not and you have different tools with you and they have
their separate jobs to execute,use them accordingly for specific use cases.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_chatbot),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm_chain = LLMChain(llm=llm, prompt=prompt)
output_parser = NoOpOutputParser()

# Initialize the agent
agent = OnboardingAgent(tools=tools, llm_chain=llm_chain, output_parser=output_parser)

# Start the conversation
print(agent.run("Let's start the onboarding process."))
while True:
    user_input = input("You: ")
    response = agent.run(user_input)
    print("Agent:", response)
