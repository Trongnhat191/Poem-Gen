import streamlit as st
from inference import DROPOUT
from inference import inference,DROPOUT,  HIDDEN_DIMS, N_HEADS, N_LAYERS, EMBEDDING_DIMS
from main import vocab, VOCAB_SIZE
from model import TransformerModel
import torch

st.title('Text Generator')

text_input = st.text_input('Enter some text')

str_text_input = str(text_input)

model = TransformerModel(VOCAB_SIZE, EMBEDDING_DIMS, N_HEADS, HIDDEN_DIMS, N_LAYERS, DROPOUT)
model.load_state_dict(torch.load('model_poem_gen_1.pth'))
model.eval()

if st.button('Enter'):
    lines = inference(model, str_text_input, vocab, 'cpu')
    for line in lines:
        line = ''.join(line)
        print(line)
        st.write(line)