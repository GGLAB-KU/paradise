import torch
import numpy as np
import json
import faiss
import argparse
import nltk
from sentence_transformers import util
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path of model")
    parser.add_argument("--warnings_file", type=str, help="Path of warnings data")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    def get_verb_and_noun_token_indexes(text, tokenizer):
        tokens = tokenizer.tokenize(text)
        verb_token_indexes = []
        for token in pos_tag(tokens):
            if (token[1] == "VB") or (token[1] == "VBD") or (token[1] == "VBG") or (token[1] == "VBN") or (token[1] == "VBP") or (token[1] == "VBZ") or (token[1] == "NN") or (token[1] == "NNP") or (token[1] == "NNS"):
                verb_token_indexes.append(tokens.index(token[0]))
        return verb_token_indexes

    def get_mean_of_the_verb_and_noun_tokens(text, verb_token_indexes, model, tokenizer):
        tokenizer_result = tokenizer(text, return_attention_mask=True, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').to(device)
        input_ids = tokenizer_result.input_ids
        attention_mask = tokenizer_result.attention_mask

        model_result = model(input_ids, attention_mask=attention_mask, return_dict=True)
        token_embeddings = model_result.last_hidden_state
        new_embeddings = torch.empty((1, 0, 768), dtype=torch.int64, device=device)
        new_attention_mask = torch.empty((1, 0), dtype=torch.int64, device=device)

        for ind in verb_token_indexes:
            if ind < 511:
                new_embeddings = torch.cat((new_embeddings, token_embeddings[:, ind+1, :].reshape(1, 1, 768)), 1)
                new_attention_mask = torch.cat((new_attention_mask, attention_mask[:, ind+1].reshape((1, 1))), 1)

        return (new_embeddings.sum(axis=1) / new_attention_mask.sum(axis=-1).unsqueeze(-1)).cpu().detach().numpy()

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    model = BertModel.from_pretrained(args.model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    warnings_file = json.load(open(args.warnings_file))
    warnings = (list(warnings_file.values()))
    warnings = [warning for warning_list in warnings for warning in warning_list]

    embeddings = []
    for warning in tqdm(warnings):
        token_indexes = get_verb_and_noun_token_indexes(warning, tokenizer)
        emb = get_mean_of_the_verb_and_noun_tokens(warning, token_indexes, model, tokenizer)
        embeddings.append(emb)
        torch.cuda.empty_cache()

    new_ndarray = np.array(embeddings)
    new_ndarray = new_ndarray.astype("float32")
    new_ndarray = new_ndarray.reshape((len(embeddings), 768))

    d = new_ndarray.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.add(new_ndarray)

    warning_inference = {}
    for warning in tqdm(warnings):
        try:
            warning_verb_token_indexes = get_verb_and_noun_token_indexes(warning, tokenizer)
            warning_embedding = get_mean_of_the_verb_and_noun_tokens(warning, warning_verb_token_indexes, model, tokenizer)

            k = 30
            D, I = index.search(warning_embedding, k)
            negative_candidates = []
            negative_candidates_similarities = []

            for i in range(k):
                token_indexes = get_verb_and_noun_token_indexes(warnings[I[0][i]], tokenizer)
                if token_indexes != []:
                    retrieved_goal_embedding = get_mean_of_the_verb_and_noun_tokens(warnings[I[0][i]], token_indexes, model, tokenizer)
                    cosine_scores = util.cos_sim(warning_embedding, retrieved_goal_embedding)
                    if (warning != warnings[I[0][i]]):
                        negative_candidates.append(warnings[I[0][i]])
                        negative_candidates_similarities.append(float(cosine_scores[0][0]))
                    torch.cuda.empty_cache()
                else:
                    pass

            negative_candidates_similarities, negative_candidates = zip(*sorted(zip(negative_candidates_similarities, negative_candidates), reverse=True))
            negative_candidates = list(negative_candidates)[:3]
            negative_candidates_similarities = list(negative_candidates_similarities)[:3]

            goal = [k for k, v in warnings_file.items() if warning in v][0]
            warning_inference[len(warning_inference)] = {"goal":goal, "positive_candidate":warning, "negative_candidates":negative_candidates, "negative_candidates_similarities":negative_candidates_similarities}
            torch.cuda.empty_cache()
        except:
            pass

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(warning_inference, f, ensure_ascii=False, indent=3)

if __name__ == "__main__":
    main()