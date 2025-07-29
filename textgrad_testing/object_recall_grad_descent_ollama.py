from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient  # (not needed anymore)
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss
import textgrad as tg
from typing import Callable, List, Any
from typing import Callable, List, Any
import os
import tqdm


# Set up LMStudio local engine
# start a server with lm-studio and point it to the right address; here we use the default address. 



def optimize_prompt_ollama(model_fn: Callable[[str, Any], Any], model_name:str, initial_prompt: str, input_set: List[Any], expected_output_set: List[Any], steps: int = 3):
    """
    Optimizes a prompt for a model function using textgrad and LMStudio so that the model's outputs on input_set match expected_output_set.
    Args:
        model_fn: Callable that takes (prompt, input) and returns model output.
        initial_prompt: The starting prompt string to optimize.
        input_set: List of inputs to feed to the model.
        expected_output_set: List of expected outputs (same length as input_set).
        steps: Number of optimization steps (default: 3).
    Returns:
        The optimized prompt string.
    """
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    engine = ChatExternalClient(client=client, model_string='ifioravanti/neuralbeagle14-7b:7b')


    assert len(input_set) == len(expected_output_set), "Input and output sets must be the same length."

    tg.set_backward_engine(engine, override=True)

    prompt = tg.Variable(value=initial_prompt, requires_grad=True, role_description="prompt to optimize")
    optimizer = tg.TGD(parameters=[prompt])

    # System prompt for feedback-based loss
    loss_system_prompt = tg.Variable(
        """You are a helpful assistant. Evaluate the model's output for the given input and expected output in the context of the prompt template. Provide concise feedback on how well the output matches the expectation, and suggest improvements for only the prompt template. Do not solve the task yourself, do not create a new prompt template yourself, only provide feedback on how to improve the prompt template to present the input values better.""",
        requires_grad=False,
        role_description="system prompt for feedback loss"
    )
    text_loss = tg.TextLoss(loss_system_prompt)
        
    for _ in range(steps):
        losses = []
        for k in range(len(input_set)):
            optimizer.zero_grad()
            inp = input_set[k]
            expected = expected_output_set[k]
            model_output = model_fn(model_name, prompt.value, inp)
            if inp == expected:
                continue
            if isinstance(model_output, list):
                temper = 0
                for i in expected:
                    if i in model_output:
                        temper += 1
                if temper == len(expected):
                    continue
            #feedback = f"Prompt Template To Improve: {prompt.value}\nInput: {inp}\nExpected: {expected}\nModel Output: {model_output}"
            feedback = f"Prompt Template To Improve: {prompt.value}\nExpected: {expected}\nModel Output: {model_output}"
            feedback_text = feedback
            print(f"Feedback Text: {feedback_text}")
            elge = tg.Variable(feedback_text, requires_grad=True, role_description="feedback for prompt")
            loss = text_loss(elge)
            losses.append(loss)
        total_loss = tg.sum(losses)
        print(f"Loss: {total_loss}")
        total_loss.backward()
        optimizer.step()
    print("Final optimized prompt:", prompt.value)

    return prompt.value

def optimize_prompt_ollama_two(model_fn: Callable[[str, Any], Any], model_name:str, initial_prompt: str, input_set: List[Any], image_input_set: List[Any], expected_output_set: List[Any], steps: int = 3):
    """
    Optimizes a prompt for a model function using textgrad and LMStudio so that the model's outputs on input_set match expected_output_set.
    Args:
        model_fn: Callable that takes (prompt, input) and returns model output.
        initial_prompt: The starting prompt string to optimize.
        input_set: List of inputs to feed to the model.
        expected_output_set: List of expected outputs (same length as input_set).
        steps: Number of optimization steps (default: 3).
    Returns:
        The optimized prompt string.
    """
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    engine = ChatExternalClient(client=client, model_string='ifioravanti/neuralbeagle14-7b:7b')


    assert len(input_set) == len(expected_output_set), "Input and output sets must be the same length."

    tg.set_backward_engine(engine, override=True)

    prompt = tg.Variable(value=initial_prompt, requires_grad=True, role_description="prompt to optimize")
    optimizer = tg.TGD(parameters=[prompt])

    # System prompt for feedback-based loss
    loss_system_prompt = tg.Variable(
        """You are a helpful assistant. Evaluate the model's output for the given input and expected output in the context of the prompt template. Provide concise feedback on how well the output matches the expectation, and suggest improvements for only the prompt template. Do not solve the task yourself, do not create a new prompt template yourself, only provide feedback on how to improve the prompt template to present the input values better.""",
        requires_grad=False,
        role_description="system prompt for feedback loss"
    )
    text_loss = tg.TextLoss(loss_system_prompt)
        
    for _ in range(steps):
        losses = []
        for k in range(len(input_set)):
            print("current prompt:", prompt.value)
            optimizer.zero_grad()
            inp = input_set[k]
            img = image_input_set[k]
            expected = expected_output_set[k]
            model_output = model_fn(model_name, prompt.value, inp, img)
            if inp == expected:
                continue
            #feedback = f"Prompt Template To Improve: {prompt.value}\nInput: {inp}\nExpected: {expected}\nModel Output: {model_output}"
            feedback = f"Prompt Template To Improve: {prompt.value}\nInput: {inp}\nExpected: {expected}\nModel Output: {model_output}"
            feedback_text = feedback
            print(f"Feedback Text: {feedback_text}")
            elge = tg.Variable(feedback_text, requires_grad=True, role_description="feedback for prompt")
            loss = text_loss(elge)
            losses.append(loss)
        total_loss = tg.sum(losses)
        print(f"Loss: {total_loss}")
        total_loss.backward()
        optimizer.step()
    print("Final optimized prompt:", prompt.value)

    return prompt.value

def optimize_prompt_ollama_image_first(model_fn: Callable[[str, Any], Any], model_name:str, initial_prompt: str, input_set: List[Any], image_input_set: List[Any], expected_output_set: List[Any], steps: int = 3):
    """
    Optimizes a prompt for a model function using textgrad and LMStudio so that the model's outputs on input_set match expected_output_set.
    Args:
        model_fn: Callable that takes (prompt, input) and returns model output.
        initial_prompt: The starting prompt string to optimize.
        input_set: List of inputs to feed to the model.
        expected_output_set: List of expected outputs (same length as input_set).
        steps: Number of optimization steps (default: 3).
    Returns:
        The optimized prompt string.
    """
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    engine = ChatExternalClient(client=client, model_string='llava:34b')


    assert len(input_set) == len(expected_output_set), "Input and output sets must be the same length."

    tg.set_backward_engine(engine, override=True)

    prompt = tg.Variable(value=initial_prompt, requires_grad=True, role_description="prompt to optimize")
    optimizer = tg.TGD(parameters=[prompt])

    # System prompt for feedback-based loss
    #loss_system_prompt = tg.Variable(
    #    """You are a helpful assistant. Evaluate the model's output for the given input and expected output in the context of the prompt template. Provide concise feedback on how well the output matches the expectation, and suggest improvements for only the prompt template. Do not solve the task yourself, do not create a new prompt template yourself, only provide feedback on how to improve the prompt template to present the input values better.""",
    #    requires_grad=False,
    #    role_description="system prompt for feedback loss"
    #)
    loss_fn = ImageQALoss(
    evaluation_instruction="""Evaluate the model's output for the given input and expected output in the context of the prompt template.
      Provide concise feedback on how well the output matches the expectation, and suggest improvements for only the prompt template. 
      Do not solve the task yourself, do not create a new prompt template yourself, 
      only provide feedback on how to improve the prompt template to present the input values better.""",
    engine=engine,
)
        
    for _ in range(steps):
        losses = []
        for k in range(len(input_set)):
            print("current prompt:", prompt.value)
            optimizer.zero_grad()
            inp = input_set[k]
            img = image_input_set[k]
            expected = expected_output_set[k]
            model_output = model_fn(model_name, prompt.value, inp, img)
            if inp == expected:
                continue
            #feedback = f"Prompt Template To Improve: {prompt.value}\nInput: {inp}\nExpected: {expected}\nModel Output: {model_output}"
            #feedback = f"Prompt Template To Improve: {prompt.value}\nInput: {inp}\nExpected: {expected}\nModel Output: {model_output}"
            #feedback_text = feedback
            #print(f"Feedback Text: {feedback_text}")
            question_variable = tg.Variable(prompt.value + " " + inp, requires_grad=False, role_description="input question")
            image_variable = tg.Variable(img, requires_grad=False, role_description="input image")
            response = tg.Variable(model_output, requires_grad=False, role_description="model response")
            loss = loss_fn(question=question_variable, image=image_variable, response=response)
            losses.append(loss)
        total_loss = tg.sum(losses)
        print(f"Loss: {total_loss}")
        total_loss.backward()
        optimizer.step()
    print("Final optimized prompt:", prompt.value)

    return prompt.value

def optimize_prompt_ollama_last_call(model_fn: Callable[[str, Any], Any], initial_prompt_set: List[str], steps: int = 3):
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    engine = ChatExternalClient(client=client, model_string='llama3:70b')

    tg.set_backward_engine(engine, override=True)

    prompt = tg.Variable(value=initial_prompt_set[0], requires_grad=True, role_description="prompt to optimize")
    optimizer = tg.TGD(parameters=[prompt])

    # System prompt for feedback-based loss
    loss_system_prompt = tg.Variable(
        """You are a helpful assistant. Evaluate the model's performance for each of the prompts and try to identify a pattern for what makes a good prompt template for this task. Provide concise feedback on what parts of the prompt contribute to a successful outcome, and suggest improvements for only the prompt template. Do not solve the task yourself, do not create a new prompt template yourself, only provide feedback on how to improve the prompt template to better achieve a result.""",
        requires_grad=False,
        role_description="system prompt for feedback loss"
    )
    text_loss = tg.TextLoss(loss_system_prompt)
        
    for _ in range(steps):
        losses = []
        feedback_text = ""
        for k in range(len(initial_prompt_set)):
            print("current prompt:", prompt.value)
            optimizer.zero_grad()
            model_output = model_fn(prompt.value)
            #feedback = f"Prompt Template To Improve: {prompt.value}\nInput: {inp}\nExpected: {expected}\nModel Output: {model_output}"
            feedback = f"Prompt Template:{prompt.value}\nModel Performance: {model_output}\n\n"
            feedback_text += feedback
            print(f"Feedback Text: {feedback_text}")
        elge = tg.Variable(feedback_text, requires_grad=True, role_description="feedback for prompt")
        loss = text_loss(elge)
        print(f"Loss: {loss}")
        loss.backward()
        optimizer.step()
    print("Final optimized prompt:", prompt.value)

    return prompt.value