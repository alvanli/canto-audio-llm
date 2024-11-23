from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel
)

if __name__ == "__main__": 
    tokenizer = AutoTokenizer.from_pretrained("hon9kon9ize/CantoneseLLMChat-v1.0-7B")
    prefix, suffix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "PLACEHOLDER"}],
        tokenize=False,
        add_generation_prompt=True,
    ).split("PLACEHOLDER")
    # print(tokenizer.encode("hello how are you"))
    print(suffix)
    # print(suffix)
    # print(tokenizer.decode([1246]))
    # print(tokenizer.decode([525]))
    # print(tokenizer.decode([498]))