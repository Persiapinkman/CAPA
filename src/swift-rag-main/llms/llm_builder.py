from llama_index.core.llms.mock import MockLLM
from llms.custom_llm import SenseNovaLLM

def build_llm(model_name,context_window,num_output,return_prompt_as_response=False,callback_manager=None):
    llm =  SenseNovaLLM(
        model_name=model_name,
        context_window=context_window,
        num_output=num_output,
        return_prompt_as_response=return_prompt_as_response,
        callback_manager=callback_manager
        )
    return llm
