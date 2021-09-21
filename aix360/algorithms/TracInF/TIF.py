from aix360.algorithms.lwbe import LocalWBExplainer

import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from transformers import \
    RobertaForSequenceClassification, AdamW

from TIF_utils import DatasetReader, training

BERT_Model = RobertaForSequenceClassification.from_pretrained(
    'roberta-large', num_labels=3, output_hidden_states=True)


class TracInFExplainer(LocalWBExplainer):
    """
    TracInfExplainer can be used to analyze sentiments in a sentence

    References:
        .. [#] `GUANGNAN YE, YADA ZHU, Zixuan Yuan, Florian Kehl`

    """
    def __init__(self, model):
        super(TracInFExplainer, self).__init__()
        self._wbmodel = model

    def set_params(self, *argv, **kwargs):
        pass

    """
    return accuracy indicator for the training of the BERT Model
    BERT Model with best accuracy is stored in model_output
    """
    def explain_instance(self, max_seq_length, BERT_name, data_dir, train_data_name, dev_data_name, batch_size,
                         epochs, model_output):

        # check if gpu processing is available
        use_gpu, device_name = False, 'cpu'
        if torch.cuda.is_available():
            device_name, use_gpu = 'cuda', True
        device = torch.device(device_name)

        # for retrain
        data_reader = DatasetReader(max_seq_length, BERT_name)
        train_dids, train_tids, train_inputs, train_masks, train_labels = data_reader.read_data(
            os.path.join(data_dir, train_data_name),
            filter=False,
            aspect_only_by_rule=False  # args.aspect_only
        )

        dev_dids, dev_tids, dev_inputs, dev_masks, dev_labels = data_reader.read_data(
            os.path.join(data_dir, dev_data_name),
            aspect_only_by_rule=False  # args.aspect_only
        )

        # read train data into code
        train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), torch.tensor(train_labels))
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # read dev data into code
        dev_data = TensorDataset(torch.tensor(dev_inputs), torch.tensor(dev_masks), torch.tensor(dev_labels))
        dev_sampler = SequentialSampler(dev_data)  # RandomSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

        print("Data prep finished")

        if use_gpu:
            BERT_Model.cuba()

        no_decay = ['bias', 'LayerNorm.weight']

        # create AdamW optimizier with focus on BERT model parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in BERT_Model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in BERT_Model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

        # train models and store most accurate model
        # return accuracy number of stored model
        return training(BERT_name, BERT_Model, epochs, train_dataloader, dev_dataloader, device, optimizer, model_output)
