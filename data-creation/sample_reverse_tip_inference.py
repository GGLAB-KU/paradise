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
    parser.add_argument("--tips_file", type=str, help="Path of tips data")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    def get_verb_and_noun_token_indexes(text, tokenizer):
        tokens = tokenizer.tokenize(text)
        verb_token_indexes = []
        for token in pos_tag(tokens):
            if (token[1] == "VB") or (token[1] == "VBD") or (token[1] == "VBG") or (token[1] == "VBN") or (token[1] == "VBP") or (token[1] == "VBZ"):
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

    tips_file = json.load(open(args.tips_file))
    goals = (list(tips_file.keys()))

    embeddings = []
    for goal in tqdm(goals):
        token_indexes = get_verb_and_noun_token_indexes(goal, tokenizer)
        emb = get_mean_of_the_verb_and_noun_tokens(goal, token_indexes, model, tokenizer)
        embeddings.append(emb)
        torch.cuda.empty_cache()

    new_ndarray = np.array(embeddings)
    new_ndarray = new_ndarray.astype("float32")
    new_ndarray = new_ndarray.reshape((len(embeddings), 768))

    d = new_ndarray.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.add(new_ndarray)

    reverse_tip_inference = {}
    for goal in tqdm(goals):
        try:
            goal_verb_token_indexes = get_verb_and_noun_token_indexes(goal, tokenizer)
            goal_embedding = get_mean_of_the_verb_and_noun_tokens(goal, goal_verb_token_indexes, model, tokenizer)

            k = 30
            D, I = index.search(goal_embedding, k)
            negative_candidates = []
            negative_candidates_similarities = []

            for i in range(k):
                token_indexes = get_verb_and_noun_token_indexes(goals[I[0][i]], tokenizer)
                if token_indexes != []:
                    retrieved_goal_embedding = get_mean_of_the_verb_and_noun_tokens(goals[I[0][i]], token_indexes, model, tokenizer)
                    cosine_scores = util.cos_sim(goal_embedding, retrieved_goal_embedding)
                    if (goal != goals[I[0][i]]):
                        negative_candidates.append(goals[I[0][i]])
                        negative_candidates_similarities.append(float(cosine_scores[0][0]))
                    torch.cuda.empty_cache()
                else:
                    pass

            negative_candidates_similarities, negative_candidates = zip(*sorted(zip(negative_candidates_similarities, negative_candidates), reverse=True))
            negative_candidates = list(negative_candidates)[:3]
            negative_candidates_similarities = list(negative_candidates_similarities)[:3]

            tips = tips_file[goal]
            reverse_tip_inference[len(reverse_tip_inference)] = {"tips": tips, "positive_candidate":goal, "negative_candidates":negative_candidates, "negative_candidates_similarities":negative_candidates_similarities}
            torch.cuda.empty_cache()
        except:
            pass

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(reverse_tip_inference, f, ensure_ascii=False, indent=3)

if __name__ == "__main__":
    main()