import torch
from transformers import AutoModelForQuestionAnswering,AutoTokenizer
import wikipedia as wiki
from transformers import AutoTokenizer,AutoModelForQuestionAnswering



def prepare_validation_features(question,context,tokenizer,pad_on_right,max_length,doc_stride):
 
    tokenized_examples = tokenizer(
        question,
        context,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        
        tokenized_examples["example_id"].append(0)


        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def convert_ids_to_string(tokenizer, input_ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))

class QAS():

    def __init__(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

        self.pad_on_right = self.tokenizer.padding_side == "right"
        self.max_length = 384
        self.doc_stride = 128
    

    def ans(self,question):

        results = wiki.search(question)
        results = wiki.page(results[0]).content

        features = prepare_validation_features(question,results,self.tokenizer,self.pad_on_right,self.max_length,self.doc_stride)
        
        if type(features['example_id']) == "int":
            l = 1
        else : l = len(features['example_id'])

        answer = []

        for i in range(l):

            inp_id = torch.LongTensor(features['input_ids'][i]).unsqueeze(0)
            attn_mask = torch.LongTensor(features['attention_mask'][i]).unsqueeze(0)
            token_id = torch.LongTensor(features['token_type_ids'][i]).unsqueeze(0)

            input = {
      "input_ids":inp_id,
      'token_type_ids':token_id,
      "attention_mask":attn_mask
            }

            ans_start,ans_end = self.model(**input,return_dict=False)

            s = torch.argmax(ans_start)
            e = torch.argmax(ans_end)+1
            ans = (self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inp_id.squeeze()[s.item():e.item()]))) 
            
            if ans != '[CLS]' and ('[CLS]' not in ans):
                answer.append(ans)
            
        return answer
            







