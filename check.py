from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel
)

if __name__ == "__main__": 
    tokenizer = AutoTokenizer.from_pretrained("hon9kon9ize/CantoneseLLMChat-v1.0-7B")
    print(tokenizer.encode("hello how are you"))
    print(tokenizer.decode([14990]))
    print(tokenizer.decode([1246]))
    print(tokenizer.decode([525]))
    print(tokenizer.decode([498]))