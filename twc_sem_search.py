import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import argparse
import json
import pdb

def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]

class SGPTQnAModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.debug = False
        print("In SGPT Q&A Constructor")


    def init_model(self,model_name = None):
        # Get our models - The package will take care of downloading the models automatically
        # For best performance: Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit
        if (self.debug):
            print("Init model",model_name)
        if (model_name is None):
            model_name = "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]

        self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]


    def tokenize_with_specb(self,texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        # Add special brackets & pay attention to them
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            if is_query:
                seq.insert(0, self.SPECB_QUE_BOS)
                seq.append(self.SPECB_QUE_EOS)
            else:
                seq.insert(0, self.SPECB_DOC_BOS)
                seq.append(self.SPECB_DOC_EOS)
            att.insert(0, 1)
            att.append(1)
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        return batch_tokens

    def get_weightedmean_embedding(self,batch_tokens, model):
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def compute_embeddings(self,input_data,is_file):
        if (self.debug):
            print("Computing embeddings for:", input_data[:20])
        model = self.model
        tokenizer = self.tokenizer

        texts = read_text(input_data) if is_file == True else input_data

        queries = [texts[0]]
        docs = texts[1:]
        query_embeddings = self.get_weightedmean_embedding(self.tokenize_with_specb(queries, is_query=True), self.model)
        doc_embeddings = self.get_weightedmean_embedding(self.tokenize_with_specb(docs, is_query=False), self.model)
        return texts,(query_embeddings,doc_embeddings)



    def output_results(self,output_file,texts,embeddings,main_index = 0):
        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        query_embeddings = embeddings[0]
        doc_embeddings = embeddings[1]
        cosine_dict = {}
        queries = [texts[0]]
        docs = texts[1:]
        if (self.debug):
            print("Total sentences",len(texts))
        for i in range(len(docs)):
            cosine_dict[docs[i]] = 1 - cosine(query_embeddings[0], doc_embeddings[i])

        if (self.debug):
            print("Input sentence:",texts[main_index])
        sorted_dict = dict(sorted(cosine_dict.items(), key=lambda item: item[1],reverse = True))
        if (self.debug):
            for key in sorted_dict:
                print("Cosine similarity with  \"%s\" is: %.3f" % (key, sorted_dict[key]))
        if (output_file is not None):
            with open(output_file,"w") as fp:
                fp.write(json.dumps(sorted_dict,indent=0))
        return sorted_dict


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='SGPT model for semantic search ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-input', action="store", dest="input",required=True,help="Input file with sentences")
        parser.add_argument('-output', action="store", dest="output",default="output.txt",help="Output file with results")
        parser.add_argument('-model', action="store", dest="model",default="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",help="model name")

        results = parser.parse_args()
        obj = SGPTQnAModel()
        obj.init_model(results.model)
        texts, embeddings = obj.compute_embeddings(results.input,is_file = True)
        results = obj.output_results(results.output,texts,embeddings)

