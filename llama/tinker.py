import torch
from generation import Llama
from tokenizer import Tokenizer

# Path to your model and tokenizer
ckpt_dir = '../llama-2-7b'
tokenizer_path = './tokenizer.model'


# Parameters for the model
max_seq_len = 512  # Sequence length for the model
max_batch_size = 1  # Batch size for inference
model_parallel_size = 1  # Adjust if using model parallelism

# Build the LLaMA instance
llama = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    model_parallel_size=model_parallel_size,
    seed=42  # Seed for reproducibility
)

# Example prompt
prompt = "Once upon a time"

# Tokenize the prompt
tokenizer = Tokenizer(model_path=tokenizer_path)
prompt_tokens = tokenizer.encode(prompt, bos=True, eos=True)

# Run the model to generate a completion
with torch.no_grad():
    generated_tokens = llama.generate(
        prompt_tokens=[prompt_tokens], 
        max_gen_len=50,  # Maximum length for generated tokens
        temperature=0.6,  # Sampling temperature
        top_p=0.9  # Nucleus sampling parameter
    )

# Decode the generated tokens back to text
completion = tokenizer.decode(generated_tokens[0][0])

print("Generated text:", completion)
