import numpy as np 
import pandas as pd 

df = pd.read_csv(r'/kaggle/input/english-to-hindi/hindi_english_parallel.csv')

(df.isnull().sum())/df.shape[0]

df = df.dropna()

x = df['hindi']
y = df['english']

pip install nltk

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

tokenized_x = x.apply(lambda sentence: word_tokenize(sentence))
tokenized_y = y.apply(lambda sentence: word_tokenize(sentence))

!pip install nltk

from collections import Counter

# Flatten all token lists into one big list
all_hi_tokens = [token for sentence in tokenized_x for token in sentence]
all_en_tokens = [token for sentence in tokenized_y for token in sentence]

# Count frequencies
hi_counter = Counter(all_hi_tokens)
en_counter = Counter(all_en_tokens)

# Define special tokens
special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

# Build vocab lists starting with special tokens
hi_vocab_list = special_tokens + sorted(hi_counter)
en_vocab_list = special_tokens + sorted(en_counter)

# Make word-to-index mappings
hi_vocab = {word: idx for idx, word in enumerate(hi_vocab_list)}
en_vocab = {word: idx for idx, word in enumerate(en_vocab_list)}

# Optional: reverse lookups too
hi_ivocab = {idx: word for word, idx in hi_vocab.items()}
en_ivocab = {idx: word for word, idx in en_vocab.items()}

def encode(tokens, vocab):
    return [vocab.get("<sos>")] + [vocab.get(tok, vocab["<unk>"]) for tok in tokens] + [vocab.get("<eos>")]


import gc
del all_hi_tokens,all_en_tokens,hi_counter,en_counter,hi_vocab_list,en_vocab_list
gc.collect()

x_ids = tokenized_x.apply(lambda tokens: encode(tokens, hi_vocab))
y_ids = tokenized_y.apply(lambda tokens: encode(tokens, en_vocab))

del tokenized_x,tokenized_y
gc.collect()

import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # [src_len, batch_size, emb_dim]
        outputs, hidden = self.gru(embedded)  # [src_len, batch_size, hidden_dim]
        return outputs, hidden
    
class DecoderGRU(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden,):
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.embedding(input)  # [1, batch_size, emb_dim]
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))  # [batch_size, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, teacher_forcing_ratio=0.1):
        batch_size = y.shape[1]
        y_len = y.shape[0]
        y_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(y_len, batch_size, y_vocab_size)

        encoder_outputs, hidden = self.encoder(x)
        input = y[0, :]  #token

        for t in range(1, y_len):
            output, hidden= self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.argmax(1)
            input = y[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

INPUT_DIM = len(hi_vocab)
OUTPUT_DIM = len(en_vocab)
EMB_DIM = 256
HID_DIM = 512

encoder = EncoderGRU(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = DecoderGRU(OUTPUT_DIM, EMB_DIM, HID_DIM)

model = Seq2Seq(encoder, decoder)


del EncoderGRU,DecoderGRU
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

del x,y,df
gc.collect()

import torch.optim as optim
PAD_IDX = en_vocab["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)


from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src = src_data
        self.trg = trg_data

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence([torch.tensor(s) for s in src_batch], padding_value=hi_vocab["<pad>"])
    trg_batch = pad_sequence([torch.tensor(t) for t in trg_batch], padding_value=en_vocab["<pad>"])
    return src_batch, trg_batch

from torch.utils.data import DataLoader

dataset = TranslationDataset(x_ids.tolist(), y_ids.tolist())
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)


del TranslationDataset,collate_fn,dataset
gc.collect

total_batches = len(loader) 
batch_iterator = iter(loader) 
epoch = 0 
total_loss = 0 
batch_iterator = iter(loader) 
for i in range(total_batches): 
    src_batch, trg_batch = next(batch_iterator) 
    src_batch = src_batch.to(device) 
    trg_batch = trg_batch.to(device) 
    optimizer.zero_grad() 
    output = model(src_batch, trg_batch) 
    output_dim = output.shape[-1] 
    output = output[1:].reshape(-1, output_dim) 
    trg = trg_batch[1:].reshape(-1) 
    loss = criterion(output, trg) 
    loss.backward() 
    optimizer.step() 
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    print(f"Epoch {epoch+1} | Loss: {loss.item() :.4f}") 
    if epoch ==10:
        break
    epoch+=1
    del trg,output,output_dim 
    gc.collect() 

sentence = "मैं स्कूल जा रहा हूँ"
tokens = word_tokenize(sentence)

# Use Hindi vocab to encode input
ids = [hi_vocab["<sos>"]] + [hi_vocab.get(tok, hi_vocab["<unk>"]) for tok in tokens] + [hi_vocab["<eos>"]]
src_tensor = torch.tensor(ids).unsqueeze(1).to(device)  # [src_len, 1]

with torch.no_grad():
    encoder_outputs, hidden = model.encoder(src_tensor)

output_tokens = []
input_token = torch.tensor([en_vocab["<sos>"]]).to(device)  # Decoder starts with English <sos>

for _ in range(50):  # Max translation length
    with torch.no_grad():
        output, hidden= model.decoder(input_token, hidden)

    top1 = output.argmax(1).item()
    if top1 == en_vocab["<eos>"]:
        break
    output_tokens.append(top1)
    input_token = torch.tensor([top1]).to(device)

# Convert English IDs back to words
prediction = [en_ivocab.get(tok, "<unk>") for tok in output_tokens]
print(" ".join(prediction))


from sklearn.model_selection import train_test_split

train_src, test_src, train_trg, test_trg = train_test_split(
    x_ids.tolist(), y_ids.tolist(), test_size=0.1, random_state=42)

def translate(sentence, model, src_vocab, trg_vocab, trg_ivocab, max_len=50, device='cpu'):
    model.eval()

    # Tokenize input sentence (assuming whitespace tokenization)
    tokens = word_tokenize(sentence)

    # Convert to source IDs
    ids = [src_vocab["<sos>"]] + [src_vocab.get(tok, src_vocab["<unk>"]) for tok in tokens] + [src_vocab["<eos>"]]
    src_tensor = torch.tensor(ids).unsqueeze(1).to(device)  # [src_len, 1]

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    input_token = torch.tensor([trg_vocab["<sos>"]]).to(device)
    output_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model.decoder(input_token, hidden)

        top1 = output.argmax(1).item()
        if top1 == trg_vocab["<eos>"]:
            break

        output_tokens.append(top1)
        input_token = torch.tensor([top1]).to(device)

    prediction = [trg_ivocab.get(tok, "<unk>") for tok in output_tokens]
    return " ".join(prediction)

hypotheses = []
references = []
count = 0
for src_seq, trg_seq in zip(test_src, test_trg):  # token ID lists
    if count == 5:
        break
    print("start")
    src_sentence = [hi_ivocab.get(tok, "<unk>") for tok in src_seq if tok not in [hi_vocab["<sos>"], hi_vocab["<eos>"], hi_vocab["<pad>"]]]
    ref_sentence = [en_ivocab.get(tok, "<unk>") for tok in trg_seq if tok not in [en_vocab["<sos>"], en_vocab["<eos>"], en_vocab["<pad>"]]]
    print("done")
    src_text = " ".join(src_sentence)
    ref_text = " ".join(ref_sentence)
    print("done")
    prediction = translate(src_text, model, hi_vocab, en_vocab, en_ivocab)  # your existing translate() function
    print("done")
    hypotheses.append(prediction)
    references.append([ref_text])
    count +=1

pip install sacrebleu

from sacrebleu import corpus_bleu

bleu = corpus_bleu(hypotheses, references)
print(f"BLEU Score: {bleu.score:.2f}")


del model
gc.collect()

# [markdown]
#  Part 2

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** 0.5  # optional: scale dot-product

    def forward(self, decoder_hidden, encoder_outputs):
    
        attn_scores = torch.sum(decoder_hidden * encoder_outputs, dim=2)  # [src_len, batch]

        #Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=0)  # [src_len, batch]

        #Compute context vector
        context = torch.sum(attn_weights.unsqueeze(2) * encoder_outputs, dim=0)  # [batch, hidden]

        return context, attn_weights 

class AttentionDecoderGRU(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.embedding(input)  # [1, batch, emb]

        context,attn_weights = self.attention(hidden, encoder_outputs)  # [batch, hidden]
        context = context.unsqueeze(0)  # [1, batch, hidden]

        rnn_input = torch.cat((embedded, context), dim=2)  # [1, batch, emb + hidden]
        output, hidden = self.gru(rnn_input, hidden)

        output = output.squeeze(0)
        context = context.squeeze(0)
        prediction = self.fc_out(torch.cat((output, context), dim=1))  # [batch, output_dim]

        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        batch_size = y.shape[1]
        y_len = y.shape[0]
        y_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(y_len, batch_size, y_vocab_size)

        encoder_outputs, hidden = self.encoder(x)
        input = y[0, :]  #token

        for t in range(1, y_len):
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = y[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

hidden_dim=512
input_dim = len(hi_vocab)
output_dim = len(en_vocab)
emb_dim = 256

attention = LuongAttention(hidden_dim)
decoder = AttentionDecoderGRU(output_dim, emb_dim, hidden_dim, attention)


model = Seq2Seq(encoder, decoder).to(device)

attention_weights = []

max_len = 100
for _ in range(max_len):
    with torch.no_grad():
        output, hidden, attn = decoder(input_token, hidden, encoder_outputs)

    attention_weights.append(attn.squeeze(1).cpu().numpy()) 


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    count = 0
    for src_batch, trg_batch in loader :
        if count == 2:
            break
        count +=1
        src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
        
        optimizer.zero_grad()
        output = model(src_batch, trg_batch)  # [trg_len, batch, vocab_size]

        # Reshape for loss
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)         # remove <sos>
        trg = trg_batch[1:].reshape(-1)                     # align targets

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        del trg,output,output_dim 
        gc.collect() 
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")


del loader
gc.collect()

def translate(sentence, model, hi_vocab, en_vocab, en_ivocab, max_len=50): 
    model.eval() 
    tokens = word_tokenize(sentence) 
    indices = [hi_vocab["<sos>"]] + [hi_vocab.get(tok, hi_vocab["<unk>"]) for tok in tokens] + [hi_vocab["<eos>"]] 
    src_tensor = torch.tensor(indices).unsqueeze(1).to(device) # [src_len, 1] 
    with torch.no_grad(): 
        encoder_outputs, hidden = model.encoder(src_tensor) 
    input_token = torch.tensor([en_vocab["<sos>"]]).to(device) 
    output_tokens = [] 
    for _ in range(max_len): 
        with torch.no_grad(): 
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs) 
        top1 = output.argmax(1).item() 
        if top1 == en_vocab["<eos>"]:
            break 
        output_tokens.append(top1) 
        input_token = torch.tensor([top1]).to(device) 
    return " ".join([en_ivocab.get(tok, "<unk>") for tok in output_tokens])

translated = translate("मेरा नाम राम है", model, hi_vocab, en_vocab, en_ivocab)
print("Predicted Translation:", translated)

translate("मैं स्कूल जा रहा हूँ", model, hi_vocab, en_vocab, en_ivocab)

translate("तुम बहुत सुंदर हो", model, hi_vocab, en_vocab, en_ivocab)

hypotheses = []
references = []
count = 0
for src_seq, trg_seq in zip(test_src, test_trg):  # token ID lists
    if count == 20:
        break
  
    src_sentence = [hi_ivocab.get(tok, "<unk>") for tok in src_seq if tok not in [hi_vocab["<sos>"], hi_vocab["<eos>"], hi_vocab["<pad>"]]]
    ref_sentence = [en_ivocab.get(tok, "<unk>") for tok in trg_seq if tok not in [en_vocab["<sos>"], en_vocab["<eos>"], en_vocab["<pad>"]]]
   
    src_text = " ".join(src_sentence)
    ref_text = " ".join(ref_sentence)
    
    prediction = translate(src_text, model, hi_vocab, en_vocab, en_ivocab)  # your existing translate() function
   
    hypotheses.append(prediction)
    references.append([ref_text])
    count +=1

from sacrebleu import corpus_bleu
bleu = corpus_bleu(hypotheses, references)
print(f"BLEU Score: {bleu.score:.2f}")

del model
gc.collect()

del translate,optimizer,criterion,decoder,encoder
gc.collect()

# [markdown]
#  Part 3

pip install transformers datasets sacrebleu

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

from datasets import Dataset, DatasetDict

# Load CSV with pandas
df = pd.read_csv('/kaggle/input/english-to-hindi/hindi_english_parallel.csv')
df = df.dropna(subset=["hindi", "english"])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into train and test sets
dataset = dataset.train_test_split(test_size=0.1) 



import gc

del df
gc.collect()

dataset = DatasetDict({
    "train": dataset["train"],
    "test": dataset["test"]
})



prefix = "translate Hindi to English: "

def preprocess(example):
    inputs = [prefix + (text if text is not None else "") for text in example["hindi"]]
    targets = [(text if text is not None else "") for text in example["english"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



tokenized = dataset.map(preprocess, batched=True)
print(dataset["train"].column_names)


import transformers
print(transformers.__version__)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt-hi-en",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer
)

trainer.train()


def translate(text):
    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True)

    # Ensure inputs go to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translate("मेरा नाम राम है।"))


pip install evaluate


import evaluate

metric = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])


import torch
from tqdm import tqdm

model.eval()

predictions = []
references = []
sources = []

for example in tqdm(tokenized["test"]):
    # Get input and target text from original columns
    src_text = example["hindi"]
    ref_text = example["english"]

    # Encode input
    inputs = tokenizer("translate Hindi to English: " + src_text, return_tensors="pt", truncation=True).to(model.device)

    # Generate prediction
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=50)
    
    pred_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Store everything
    predictions.append(pred_text)
    references.append([ref_text])  # wrap in list for sacrebleu format
    sources.append(src_text)


import evaluate

bleu = evaluate.load("sacrebleu")
result = bleu.compute(predictions=predictions, references=references)

print(f"BLEU score: {result['score']:.2f}")


