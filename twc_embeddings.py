import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import argparse
import json
import pdb

def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]


class SGPTModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.debug = False
        print("In SGPT Constructor")


    def init_model(self,model_name = None):
        # Get our models - The package will take care of downloading the models automatically
        # For best performance: Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit
        if (self.debug):
            print("Init model",model_name)
        if (model_name is None):
            model_name = "Muennighoff/SGPT-125M-weightedmean-nli-bitfit"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        #self.tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit")
        #self.model = AutoModel.from_pretrained("Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit")
        #self.tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit")
        #self.model = AutoModel.from_pretrained("Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit")
        # Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
        self.model.eval()

    def compute_embeddings(self,input_file_name,input_data,is_file):
        if (self.debug):
            print("Computing embeddings for:", input_data[:20])
        model = self.model
        tokenizer = self.tokenizer

        texts = read_text(input_data) if is_file == True else input_data

        # Tokenize input texts
        batch_tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

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
        return texts,embeddings

    def output_results(self,output_file,texts,embeddings,main_index = 0):
        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        cosine_dict = {}
        if (self.debug):
            print("Total sentences",len(texts))
        for i in range(len(texts)):
                cosine_dict[texts[i]] = 1 - cosine(embeddings[main_index], embeddings[i])

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
        parser = argparse.ArgumentParser(description='SGPT model for sentence embeddings ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-input', action="store", dest="input",required=True,help="Input file with sentences")
        parser.add_argument('-output', action="store", dest="output",default="output.txt",help="Output file with results")
        parser.add_argument('-model', action="store", dest="model",default="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",help="model name")

        results = parser.parse_args()
        obj = SGPTModel()
        obj.init_model(results.model)
        texts, embeddings = obj.compute_embeddings(results.input,results.input,is_file = True)
        results = obj.output_results(results.output,texts,embeddings)
