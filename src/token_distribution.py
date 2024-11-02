import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def check_for_word(tokenizer, text):
    """
    Check if a word is in the model's vocabulary
    """
    vocab = tokenizer.vocab
    try:
        return vocab[text]
    except:
        return "Word not in vocabulary"

def generate_prompt(tokenizer, system_prompt, user_prompt):
    """
    Formats a prompt for Qwen2.5 model
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        return "Error with prompt template, check huggingface docs"
    
def get_top_k_words(model, tokenizer, text, top_k=3):
    """
    Get the top k words with the highest probabilities for a given text
    """
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model(**model_inputs) #don't use the generate function here to be able to see the logits
    last_token_logits = outputs.logits[:, -1, :] #[batch_size, sequence_length, vocab_size]
    probabilities = F.softmax(last_token_logits, dim=-1) #convert logits to probabilities with softmax
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=1)#get the probability values and indicies
    top_k_words = [tokenizer.decode(idx) for idx in top_k_indices[0]] #pull the indices from 2D tensor to 1D tensor to be treated as a list    

    for word, prob in zip(top_k_words, top_k_probs[0].tolist()):
        print(f"{word}: {prob:.4f}")
