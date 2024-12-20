from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel
)

if __name__ == "__main__": 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    # print(tokenizer.decode([128000, 128006, 882, 128007, 271]))
    # print(tokenizer.decode([128009, 128006, 78191, 128007, 271]))
    # prefix, suffix = tokenizer.apply_chat_template(
    #     [{"role": "user", "content": "PLACEHOLDER"}],
    #     tokenize=False,
    #     add_generation_prompt=True,
    # ).split("PLACEHOLDER")
    # # print(tokenizer.encode("hello how are you"))
    # print(suffix)
    # print(suffix)
    # print(tokenizer.decode([1246]))
    # print(tokenizer.decode([525]))
    # print(tokenizer.decode([498]))