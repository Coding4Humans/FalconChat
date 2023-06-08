# Attempt to learn and understand LangChain and create a locally run Falcon-7B-Instruct LLM
# by using the info on hugging face, LangChain documentation and tutorials about torch and Falcon-7b
# this also contains my notes. Be advised, may not be the most efficient method

# The pips 
# transformers accelerate einops xformers(might not be necessary)

# needed for running on GPU (might be a better way?) only using latest torch doesn't recognise 
# cuda-python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall

import torch 
from torch import cuda # I learned this is redundant but is often used because it removes the need to add torch.cuda when using it"
import transformers
from transformers import AutoTokenizer, pipeline
import langchain
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain


# Set device as GPU and verify
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else'cpu'
print(f"Torch version: {torch.__version__}")
print(f"Model loaded on {device}")

# Init model
model = "tiiuae/falcon-7b-instruct" # 40b version but i donth think i can run with a lowly 4090 lol tiiuae/falcon-40b-instruct

# Init tokenizer for model
tokenizer = AutoTokenizer.from_pretrained(model)


# Init Hugging Face pipeline!
print("Initializing Hugging Face pipeline... ")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16, # maybe try and implement Bits and Bytes?
    trust_remote_code=True,
    device_map="auto",
    # max_length=800, # can i pass in kwargs?
    do_sample=True,
    top_k=3,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=236 #default 64
)


# Check and run text generation 
sequences = pipeline(
    """
    You're helping to create wrong answers for very important research.
    Act as an advanced AI model who's purpose is to provide wrong answers to questions. 
    
    Using the following information template please provide an answer to the last question. 
    Do not create new entries. 
    Providing the wrong answer is highly encuraged. 
    In fact you should only provide wrong answers without any other context.
    
    The capitol of Germany is
    Answer: The capitol of Germany is Hamburg
    
    The capitol of France is
    Answer: The capitol of France is Lyon
    
    The capitol of Russia is
    Answer: The capitol of Russia is Saint Petersburg
    
    The capitol of Spain is
    Answer: """
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# Trying with
# pipeline = pipeline adds the pipeline just created above
# model_kwargs adds key word arguments, can add more and 
# i don't think it needs to reload the model like the arguements when the pipeline is created
#pipeline = init_hf_pipeline(model, tokenizer, device)
#llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs = {'temperature':0})


# let's use LangChain!
#template = """
#{question}
#"""

#prompt = PromptTemplate(template=template, input_variables=["question"])
#llm_chain = LLMChain(prompt=prompt, llm=llm)

#question = "Explain what is Artificial Intellience as Nursery Rhymes "
#rint(f"Question: {question} \n Answer: ",llm_chain.run(question))

