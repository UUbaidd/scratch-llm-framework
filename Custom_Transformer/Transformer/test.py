import numpy as np
from custom_dl.Tokenizer import BPETokenizer
from custom_dl.Models import LanguageModel

# 1. Initialize custom Tokenizer
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6}
tokenizer = BPETokenizer()

# Let's use 5 merges for a clean split
tokenizer.fit(vocab, merges=5)  

# --- THE FIX: Stop guessing! Let's print exactly what survived ---
print("--- TOKENIZER STATE ---")
print("Learned Tokens in Dictionary:", list(tokenizer.token_to_id.keys()))

# 2. Build Model using clean dimensions
D_MODEL = 8
HEADS = 2
VOCAB_SIZE = len(tokenizer.token_to_id)
model = LanguageModel(VOCAB_SIZE, D_MODEL, HEADS)

# 3. Process Sequence
# Instead of us typing a string manually, we will pull the exact 
# mutated 'newest' string directly from the tokenizer's final state.
# (The 'newest' sequence is the 3rd item in our vocab dict)
example_text = list(tokenizer.vocab.keys())[2] 
print(f"\nUsing perfectly matched token string: '{example_text}'")

# Map tokens safely to IDs. This will NEVER throw a KeyError now.
input_ids = [tokenizer.token_to_id[t] for t in example_text.split()]

# 4. Pass it to your Language Model
probabilities = model(input_ids)
print("\n--- MODEL EXECUTION ---")
print(f"Success! Output Probabilities Shape: {probabilities.shape}")

# 5. Extract predictions
last_token_probs = probabilities[0, -1, :]
predicted_id = np.argmax(last_token_probs)

# 6. Decode back to readable text safely
predicted_word = tokenizer.id_to_token[predicted_id]

print(f"Input Tokens: {input_ids}")
print(f"Model predicts the next token ID is: {predicted_id} ('{predicted_word}')")