from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain_core.prompts.prompt import PromptTemplate
from colorama import Fore
import os
import warnings

warnings.filterwarnings("ignore")

load_dotenv(find_dotenv())

# os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

template = """The following is a friendly conversation between a human and an AI 
If the AI does not know the answer to a question, it truthfully says it does not 
know. The AI answers in one or two sentences

Current conversation:
{history}
Human: {input}
AI Assistant:"""

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "your are this personal friend"}, {"output": "hi"})
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

conversation = ConversationChain(llm=llm, memory=memory, verbose=False,prompt=PROMPT)
if __name__ == "__main__":
    try:
        os.system("cls")
        print(Fore.GREEN+"#####Begin Chatting with your AI friend####")        
        print(Fore.RED+"press Crtl+c to exit")
        while True:
            a = input(Fore.BLUE+ "[+]You:")
            answer = conversation.predict(input=a)
            print(Fore.GREEN+"[+]AI:"+answer )
    except KeyboardInterrupt as e:
        print(Fore.RED+"\nExiting now ...")
