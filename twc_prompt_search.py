import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cosine

# Get models - The package will take care of downloading the models automatically
# For best performance: EleutherAI/gpt-j-6B
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
# Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
model.eval()

prompt = 'Documents are searched to find matches with the same content.\nThe document "{}" is a good search result for "'

queries = [
    "I'm searching for a planet not too far from Earth.",
]


docs = [
    "Mercury is closest to the sun",
    "Venus and Mars are closest to earth",
    "Pluto is not too far from neptune",
    "Neptune is the eighth and farthest-known Solar planet from the Sun. In the Solar System, it is the fourth-largest planet by diameter, the third-most-massive planet, and the densest giant planet. It is 17 times the mass of Earth, slightly more massive than its near-twin Uranus.",
    "TRAPPIST-1d, also designated as 2MASS J23062928-0502285 d, is a small exoplanet (about 30% the mass of the earth), which orbits on the inner edge of the habitable zone of the ultracool dwarf star TRAPPIST-1 approximately 40 light-years (12.1 parsecs, or nearly 3.7336×1014 km) away from Earth in the constellation of Aquarius.",
    "A harsh desert world orbiting twin suns in the galaxy’s Outer Rim, Tatooine is a lawless place ruled by Hutt gangsters. Many settlers scratch out a living on moisture farms, while spaceport cities such as Mos Eisley and Mos Espa serve as home base for smugglers, criminals, and other rogues.",
    "For some individual rights are very close to the heart and the decimation of it was uncacceptable",
    "The asteriod fell not too far from North Dakota",
    "The search for the missing person yielded a body not too far from the lake",
]
for query in queries:
    print(f"Query: {query}")
    for doc in docs:
        context = prompt.format(doc)

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
        # The higher (closer to 0), the more similar
        print(f"Document: {doc[:100] + '...'} Score: {score}")
