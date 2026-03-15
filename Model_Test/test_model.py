import spacy
import numpy as np
import pandas as pd
from spacy.training import offsets_to_biluo_tags
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification
from seqeval.metrics import classification_report, accuracy_score, f1_score
# Load spaCy model
nlp = spacy.load("en_core_web_sm")
# Adding '\n' to the default spacy tokenizer
prefixes = ('\n', ) + tuple(nlp.Defaults.prefixes)
prefix_regex = spacy.util.compile_prefix_regex(prefixes)
nlp.tokenizer.prefix_search = prefix_regex.search
# Personal Custom Tags Dictionary
entity_dict = {
    'Name': 'NAME', 
    'College Name': 'CLG',
    'Degree': 'DEG',
    'Graduation Year': 'GRADYEAR',
    'Years of Experience': 'YOE',
    'Companies worked at': 'COMPANY',
    'Designation': 'DESIG',
    'Skills': 'SKILLS',
    'Location': 'LOC',
    'Email Address': 'EMAIL'
}
# Define tag2idx mapping (same as in training)
tag_vals = {'X', '[CLS]', '[SEP]', 'O', 'B-NAME', 'I-NAME', 'L-NAME', 'U-NAME', 
            'B-CLG', 'I-CLG', 'L-CLG', 'U-CLG', 'B-DEG', 'I-DEG', 'L-DEG', 'U-DEG',
            'B-GRADYEAR', 'I-GRADYEAR', 'L-GRADYEAR', 'U-GRADYEAR', 'B-YOE', 'I-YOE', 'L-YOE', 'U-YOE',
            'B-COMPANY', 'I-COMPANY', 'L-COMPANY', 'U-COMPANY', 'B-DESIG', 'I-DESIG', 'L-DESIG', 'U-DESIG',
            'B-SKILLS', 'I-SKILLS', 'L-SKILLS', 'U-SKILLS', 'B-LOC', 'I-LOC', 'L-LOC', 'U-LOC',
            'B-EMAIL', 'I-EMAIL', 'L-EMAIL', 'U-EMAIL'}
tag2idx = {t: i for i, t in enumerate(tag_vals)}
idx2tag = {tag2idx[key]: key for key in tag2idx.keys()}
# Function to merge overlapping intervals
def mergeIntervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                if lower[2] is higher[2]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound, lower[2])
                else:
                    if lower[1] > higher[1]:
                        merged[-1] = lower
                    else:
                        merged[-1] = (lower[0], higher[1], higher[2])
            else:
                merged.append(higher)
    return merged

# Extract entities from annotations
def get_entities(df):
    entities = []
    for i in range(len(df)):
        entity = []
        if df['annotation'][i] is not None:
            for annot in df['annotation'][i]:
                try:
                    ent = entity_dict[annot['label'][0]]
                    start = annot['points'][0]['start']
                    end = annot['points'][0]['end'] + 1
                    entity.append((start, end, ent))
                except:
                    pass
            entity = mergeIntervals(entity)
            entities.append(entity)
        else:
            entities.append([])
    return entities

# Prepare test data
def get_test_data(df):
    tags = []
    sentences = []
    for i in range(len(df)):
        text = df['content'][i]
        entities = df['entities'][i]
        doc = nlp(text)
        tag = offsets_to_biluo_tags(doc, entities)
        tmp = pd.DataFrame([list(doc), tag]).T
        loc = []
        for i in range(len(tmp)):
            if tmp[0][i].text == '.' and tmp[1][i] == 'O':
                loc.append(i)
        loc.append(len(doc))
        last = 0
        data = []
        for pos in loc:
            data.append([list(doc)[last:pos], tag[last:pos]])
            last = pos
        for d in data:
            tag = ['O' if t == '-' else t for t in d[1]]
            if len(set(tag)) > 1:
                sentences.append(d[0])
                tags.append(tag)
    return sentences, tags

# Tokenize test data
def get_tokenized_test_data(sentences, tags):
    tokenized_texts = []
    word_piece_labels = []
    for word_list, label in zip(sentences, tags):
        temp_lable = ['[CLS]']
        temp_token = ['[CLS]']
        for word, lab in zip(word_list, label):
            token_list = tokenizer.tokenize(word.text)
            for m, token in enumerate(token_list):
                temp_token.append(token)
                if m == 0:
                    temp_lable.append(lab)
                else:
                    temp_lable.append('X')
        temp_lable.append('[SEP]')
        temp_token.append('[SEP]')
        tokenized_texts.append(temp_token)
        word_piece_labels.append(temp_lable)
    return tokenized_texts, word_piece_labels

# Convert predictions to original format
def convert_predictions(predictions, word_piece_labels, tokenized_texts):
    pred_tags = []
    true_tags = []
    for i, pred in enumerate(predictions):
        pred_tag = []
        true_tag = []
        for j, p in enumerate(pred):
            if word_piece_labels[i][j] != 'X' and word_piece_labels[i][j] != '[CLS]' and word_piece_labels[i][j] != '[SEP]':
                pred_tag.append(idx2tag[p])
                true_tag.append(word_piece_labels[i][j])
        pred_tags.append(pred_tag)
        true_tags.append(true_tag)
    return pred_tags, true_tags

# Main testing function
def test_model():
    print("Loading test data...")
    # Load test data
    test_df = pd.read_json('./test.json')
    # Drop unnecessary columns
    if 'extras' in test_df.columns:
        test_df = test_df.drop(['extras'], axis=1)
    if 'metadata' in test_df.columns:
        test_df = test_df.drop(['metadata'], axis=1)
    # Extract entities
    test_df['entities'] = get_entities(test_df)
    # Prepare test data
    test_sentences, test_tags = get_test_data(test_df)
    print(f"Processed {len(test_sentences)} test sentences")
    # Tokenize test data
    tokenized_texts, word_piece_labels = get_tokenized_test_data(test_sentences, test_tags)
    # Convert to input format
    MAX_LEN = 512
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels], 
                         maxlen=MAX_LEN, value=tag2idx["O"], padding="post", 
                         dtype="long", truncating="post")
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    # Convert to PyTorch tensors
    test_inputs = torch.tensor(input_ids)
    test_tags = torch.tensor(tags)
    test_masks = torch.tensor(attention_masks)
    # Create DataLoader
    test_data = TensorDataset(test_inputs, test_masks, test_tags)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8)
    # Load model
    print("Loading model from fifth_epoch.pt...")
    model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2idx))
    model.load_state_dict(torch.load("fifth_epoch.pt"))
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Evaluation mode
    model.eval()
    # Tracking variables
    predictions = []
    true_labels = []
    eval_loss = 0
    nb_eval_steps = 0
    print("Starting evaluation...")
    # Evaluate data
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, len(tag2idx)), b_labels.view(-1))
        eval_loss += loss.item()
        nb_eval_steps += 1
        # Get predictions
        preds = torch.argmax(logits, dim=2)
        # Convert predictions to CPU
        predictions.extend([p.detach().cpu().numpy() for p in preds])
        true_labels.extend([t.detach().cpu().numpy() for t in b_labels])
    # Calculate average loss
    eval_loss = eval_loss / nb_eval_steps
    print(f"Test Loss: {eval_loss:.4f}")
    # Convert predictions to tags
    pred_tags, true_tags = convert_predictions(predictions, word_piece_labels, tokenized_texts)
    # Calculate metrics
    accuracy = accuracy_score(true_tags, pred_tags)
    f1 = f1_score(true_tags, pred_tags)
    report = classification_report(true_tags, pred_tags)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(report)
    # Extract entities from test data
    print("\nSample Predictions:")
    for i in range(min(5, len(test_sentences))):
        tokens = [token.text for token in test_sentences[i]]
        print(f"Text: {' '.join(tokens)}")
        print(f"True: {true_tags[i]}")
        print(f"Pred: {pred_tags[i]}")
        print("-" * 50)
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    # Run test
    test_model()