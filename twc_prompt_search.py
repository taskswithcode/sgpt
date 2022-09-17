import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cosine
import argparse
import json
import pdb

def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]

class CausalLMModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.debug = False
        print("In CausalLMModel Constructor")

    def init_model(self,model_name = None):
        # Get our models - The package will take care of downloading the models automatically
        # For best performance: Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit
        if (self.debug):
            print("Init model",model_name)
        # For best performance: EleutherAI/gpt-j-6B
        if (model_name is None):
            model_name = "EleutherAI/gpt-neo-125M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.prompt = 'Documents are searched to find matches with the same content.\nThe document "{}" is a good search result for "'

    def compute_embeddings(self,input_data,is_file):
        if (self.debug):
            print("Computing embeddings for:", input_data[:20])
        model = self.model
        tokenizer = self.tokenizer

        texts = read_text(input_data) if is_file == True else input_data
        query = texts[0]
        docs = texts[1:]

        # Tokenize input texts

        #print(f"Query: {query}")
        scores = []
        for doc in docs:
            context = self.prompt.format(doc)

            context_enc = tokenizer.encode(context, add_special_tokens=False)
            continuation_enc = tokenizer.encode(query, add_special_tokens=False)
            # Slice off the last token, as we take its probability from the one before
            model_input = torch.tensor(context_enc+continuation_enc[:-1])
            continuation_len = len(continuation_enc)
            input_len, = model_input.shape

            # [seq_len] -> [seq_len, vocab]
            logprobs = torch.nn.functional.log_softmax(model(model_input)[0], dim=-1).cpu()
            # [seq_len, vocab] -> [continuation_len, vocab]
            logprobs = logprobs[input_len-continuation_len:]
            # Gather the log probabilities of the continuation tokens -> [continuation_len]
            logprobs = torch.gather(logprobs, 1, torch.tensor(continuation_enc).unsqueeze(-1)).squeeze(-1)
            score = torch.sum(logprobs)
            scores.append(score.tolist())
        return texts,scores

    def output_results(self,output_file,texts,scores,main_index = 0):
        cosine_dict = {}
        docs = texts[1:]
        if (self.debug):
            print("Total sentences",len(texts))
        assert(len(scores) == len(docs))
        for i in range(len(docs)):
            cosine_dict[docs[i]] = scores[i]

        if (self.debug):
            print("Input sentence:",texts[main_index])
        sorted_dict = dict(sorted(cosine_dict.items(), key=lambda item: item[1],reverse = True))
        if (self.debug):
            for key in sorted_dict:
                print("Document score for \"%s\" is: %.3f" % (key[:100], sorted_dict[key]))
        if (output_file is not None):
            with open(output_file,"w") as fp:
                fp.write(json.dumps(sorted_dict,indent=0))
        return sorted_dict


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='EleutherAI model used for semantic search ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-input', action="store", dest="input",required=True,help="Input file with sentences")
        parser.add_argument('-output', action="store", dest="output",default="output.txt",help="Output file with results")
        parser.add_argument('-model', action="store", dest="model",default="EleutherAI/gpt-neo-125M",help="model name")

        results = parser.parse_args()
        obj = CausalLMModel()
        obj.init_model(results.model)
        texts, embeddings = obj.compute_embeddings(results.input,is_file = True)
        results = obj.output_results(results.output,texts,embeddings)
