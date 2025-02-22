import torch
from model import TransformerModel
import torch.nn.functional as F
from PoemDataset import PoemDataset
from main import tokenizer, decode, build_vocab
import pandas as pd

DATASET_PATH = 'poem_dataset_final/poem_final.csv'
EMBEDDING_DIMS = 128
HIDDEN_DIMS = 128
N_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.2

def sample_with_temperature(logits, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature

    probabilities = F.softmax(logits, dim=-1)

    sampled_index = torch.multinomial(probabilities, 1).item()

    return sampled_index

def inference(model, input_text, vocab, device, temperature=1.2):
    model.eval()
    input_tokens = tokenizer(input_text)
    input_ids = [vocab[token] for token in input_tokens]
    eos_token_id = vocab['<eos>']
    generated_ids = input_ids.copy()
    MAX_GENERATION_LEN = 50
    for _ in range(MAX_GENERATION_LEN):
        input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)

        last_token_logits = outputs[0, -1, :]
        next_token_id = sample_with_temperature(last_token_logits, temperature)
        generated_ids.append(next_token_id)

        if next_token_id == eos_token_id:
            break

    # Convert the generated tokens back to text
    generated_text = decode(generated_ids, vocab)
    generated_text = ' '.join(generated_text)
    generated_text = generated_text.replace('<sos>', '')
    lines = generated_text.split('<eol>')
    return lines

if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    vocab = build_vocab(df)
    VOCAB_SIZE = len(vocab)

    model = TransformerModel(VOCAB_SIZE, EMBEDDING_DIMS, N_HEADS, HIDDEN_DIMS, N_LAYERS, DROPOUT)
    model.load_state_dict(torch.load('model_poem_gen.pth'))
    model.eval()
    lines = inference(model, 'The sky is', vocab, 'cpu')
    for line in lines:
        print(''.join(line))