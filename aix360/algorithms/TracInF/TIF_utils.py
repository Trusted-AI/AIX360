from transformers import BertTokenizer, RobertaTokenizer
import json
import torch
from collections import Counter
from tqdm import trange
import numpy as np
debug = False


class DatasetReader():

    def __init__(self, max_len, BERT_name):
        self.max_len = max_len
        self.bertname = BERT_name

    def read_data(self,
                  filename,
                  aspect_only_by_rule=False,
                  mask_version=False,
                  return_origin=False,
                  filter=False,
                  id_filter_list=None,
                  is_sentihood=False,
                  is_test=False,
                  select_test_list=None
                  ):
        dj = data_loader(filename)
        if is_sentihood: label_map = {'positive':1, 'negative':0}
        else: label_map = {'positive': 2, 'neutral': 1, 'negative': 0}

        if filter:
            if is_test:
                pairs = [(int(did), int(tid), data['original_text'], term['term'], term['answers'],
                          label_map[term['polarity']])
                         for did, data in dj.items()
                         for tid, term in data['terms'].items() if
                         (int(did), int(tid)) in select_test_list]
            else:
                pairs = [(int(did), int(tid), data['original_text'], term['term'], term['answers'],
                          label_map[term['polarity']])
                         for did, data in dj.items()
                         for tid, term in data['terms'].items() if
                         (int(did), int(tid)) not in id_filter_list]
        else:
            pairs = [
                (int(did), int(tid), data['original_text'], term['term'], term['answers'], label_map[term['polarity']])
                for did, data in dj.items()
                for tid, term in data['terms'].items()
            ]
        dids = [did for (did, tid, a, t, an, b) in pairs]
        tids = [tid for (did, tid, a, t, an, b) in pairs]
        origin_sentences = [a for (did, tid, a, t, an, b) in pairs]
        terms = [t for (did, tid, a, t, an, b) in pairs]
        labels = [b for (did, tid, a, t, an, b) in pairs]

        print('Data Size is: {0}, It is mask version? {1}, label bias: {2}'.format(len(labels), mask_version,
                                                                                   Counter(labels)))
        if self.bertname == 'bert-base-uncased':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            sentences = [
                '[CLS] ' + t + ' [SEP] ' + self.mask_tokens(s, an, tokenizer, t, mask_version, '[MASK]', '[CLS]',
                                                            '[SEP]') + ' [SEP] ' for (did, tid, s, t, an, b) in pairs]
        elif self.bertname == 'roberta-large':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
            sentences = []
            for did, tid, s, t, an, b in pairs:
                masked_sentence = self.mask_tokens(s, an, tokenizer, t, mask_version, '<mask>', '<s>', '</s>',
                                                   start_symbol='Ġ')
                if masked_sentence:
                    sentence = '<s> ' + t + ' </s> <s> ' + masked_sentence + ' </s> '
                else:
                    sentence = '<s> ' + '<mask>' + ' </s> <s> ' + s + ' </s> '
                sentences.append(sentence)

        encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)

        if return_origin:
            return dids, tids, encoding['input_ids'], encoding['attention_mask'], labels, [dids, tids, origin_sentences,
                                                                                           terms, labels]

        return dids, tids, encoding['input_ids'], encoding['attention_mask'], labels

    def mask_tokens(self, sentence, answers, tokenizer, term, to_mask, mask_token, start_token, end_token,
                    start_symbol=''):

        # if not to mask, return sentence itself.
        if not to_mask: return sentence

        tokens = tokenizer.tokenize(sentence)
        tokens = [token.strip(start_symbol) for token in tokens]
        tokenized_sentence = ' '.join(tokens)

        valid_answers = [ans for aid, ans in answers[0].items()
                         if aid not in 'rules'
                         and start_token not in ans
                         and end_token not in ans
                         and len(ans) > 0
                         ]

        if len(valid_answers) == 0:
            # if no answer, mask tokens between close split punctuations
            puncsidx = [i for i, token in enumerate(tokens) if token in set([',', '.', '?', '!', '--'])]
            termidx = [i for i, token in enumerate(tokens) if token == term]
            if len(termidx) == 0:
                if debug: print('no term matched. return whole sentence.')
                return None  # can not find term in sentence. Skip it.
            if len(puncsidx) == 0:
                if debug: print('Term matched but no punc. Mask till end of sentence.')
                return ' '.join([token for i, token in enumerate(tokens) if
                                 i > max(termidx)])  # mask words that are after the last matched term.
            maskposs = []
            for tid in termidx:
                # search to right for punc
                endpos = [pid for pid in puncsidx if pid > tid]
                if len(endpos) == 0:
                    endpos = len(tokens)
                else:
                    endpos = min(endpos)
                # search backward for punc
                stpos = [pid for pid in puncsidx if pid < tid]
                if len(stpos) == 0:
                    stpos = 0
                else:
                    stpos = max(stpos)
                maskposs.append((stpos, endpos))
            for stpos, endpos in maskposs:
                tokens = [mask_token if i >= stpos and i < endpos else token for i, token in enumerate(tokens)]
            if debug: print('no answer matched, but heuristically matched a span around term.')
            return ' '.join(tokens)

        answers = Counter(valid_answers)
        matched = None
        for ans, freq in answers.most_common():
            stpos = tokenized_sentence.find(ans)
            if stpos == -1:
                breakpoint()
            edpos = stpos + len(ans)
            if debug: print('answer whole string matched')
            return tokenized_sentence[:stpos] + ' ' + ' '.join([mask_token] * len(ans.split(' '))) + ' ' + sentence[
                                                                                                           edpos:]

        # mask individual words instead.
        for ans, freq in answers.most_common():
            anstoks = {a: 0 for a in ans.strip().split(' ')}
            if debug: print('answer per-term matched ')
            return ' '.join([mask_token if token in anstoks else token for token in tokens])

    # Deprecated
    def mask(self, sentence, answers, tokenizer, term, to_mask, mask_token, start_token, end_token):
        if not to_mask:
            return sentence
        valid_answers = [ans for aid, ans in answers[0].items()
                         if aid not in 'rules'
                         and start_token not in ans
                         and end_token not in ans
                         and len(ans) > 0
                         ]
        answers = Counter(valid_answers)
        matched = None
        for ans, freq in answers.most_common():
            matched = self.fuzzymatch(sentence, ans, mask_token)
            if matched:
                print(f'span found! term: {term} - answer:{ans} -sentence:{sentence}')
                return matched

        print(f'span not found from all answers!! term: {term} -sentence:{sentence}')
        return self.span_mask(sentence, term, mask_token)


def data_loader(filename):
    data = json.load(open(filename, 'r'))
    return data


def save_model(net, path):
    torch.save(net.state_dict(), path)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def training(BERT_name, BERT_Model, epochs, train_dataloader, dev_dataloader, device, optimizer, model_output):
    train_loss_set = []
    global_eval_accuracy = 0
    print("Starting training")

    # fine tune only last layer and output layer.
    for name, param in BERT_Model.base_model.named_parameters():
        param.requires_grad = False
        if 'encoder.layer.11' in name or 'encoder.layer.10' in name or 'pooler.dense' in name:
            param.requires_grad = True
    for epoch in trange(epochs, desc="Epoch"):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        BERT_Model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print()
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            print("\rEpoch:", epoch, "step:", step, end='')
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = BERT_Model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels)
            # Bert Model returns a certain loss
            train_loss_set.append(float(outputs.loss))
            # Backward pass / backward gradient descent
            outputs.loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += float(outputs.loss)
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps / nb_tr_examples))

        ##### Validation #####
        # calculate overall accuracy in the dev dataset
        # Put model in evaluation mode to evaluate loss on the validation set
        BERT_Model.eval()
        # Tracking variables
        global_eval_accuracy = 0
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = BERT_Model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        # store best BERT model
        if eval_accuracy > global_eval_accuracy:
            global_eval_accuracy = eval_accuracy
            save_model(BERT_Model, BERT_name + '-' + model_output)

    return global_eval_accuracy
