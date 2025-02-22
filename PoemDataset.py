from torch.utils.data import Dataset
import torch

class PoemDataset(Dataset):
    def __init__(self, df, tokenizer, vectorizer, max_seq_len, PAD_TOKEN):
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.PAD_TOKEN = PAD_TOKEN
        self.max_seq_len = max_seq_len
        self.input_seqs, self.target_seqs, self.padding_masks = self.create_samples(df)

    def create_padding_mask(self, input_ids):
        return [0 if token_id == self.PAD_TOKEN else 1 for token_id in input_ids]

    def split_content(self, content):
        samples = []

        poem_parts = content.split('\n\n')
        for poem_part in poem_parts:
            poem_in_lines = poem_part.split('\n')
            if len(poem_in_lines) == 4:
                samples.append(poem_in_lines)

        return samples

    def prepare_sample(self, sample):
        input_seqs = []
        target_seqs = []
        padding_masks = []

        input_text = '<sos> ' + ' <eol> '.join(sample) + ' <eol> <eos>'
        input_ids = self.tokenizer(input_text)
        for idx in range(1, len(input_ids)):
            input_seq = ' '.join(input_ids[:idx])
            target_seq = ' '.join(input_ids[1:idx+1])
            input_seq = self.vectorizer(input_seq, self.max_seq_len)
            target_seq = self.vectorizer(target_seq, self.max_seq_len)
            padding_mask = self.create_padding_mask(input_seq)

            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
            padding_masks.append(padding_mask)

        return input_seqs, target_seqs, padding_masks

    def create_samples(self, df):
        input_seqs = []
        target_words = []
        padding_masks = []

        for idx, row in df.iterrows():
            content = row['content']
            samples = self.split_content(content)
            for sample in samples:
                sample_input_seqs, sample_target_words, sample_padding_masks = self.prepare_sample(sample)

                input_seqs += sample_input_seqs
                target_words += sample_target_words
                padding_masks += sample_padding_masks

        input_seqs = torch.tensor(input_seqs, dtype=torch.long)
        target_words = torch.tensor(target_words, dtype=torch.long)
        padding_masks = torch.tensor(padding_masks, dtype=torch.float)

        return input_seqs, target_words, padding_masks

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        input_seqs = self.input_seqs[idx]
        target_seqs = self.target_seqs[idx]
        padding_masks = self.padding_masks[idx]

        return input_seqs, target_seqs, padding_masks

