import argparse 
import torch
import json
import random
from filters import tf_idf_filter, category_filter, lexical_overlap_filter, length_filter, similarity_filter, source_tutorial_filter, similarity_filter_for_pos_neg_candidates
from transformers import AutoTokenizer
from tqdm import tqdm
from simcse import SimCSE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path of tokenizer")
    parser.add_argument("--warnings_inference_file", type=str, help="Path of warnings data")
    parser.add_argument("--warnings_file", type=str, help="Path of warnings data")
    parser.add_argument("--wikihow_file", type=str, help="Path of warnings data")
    parser.add_argument("--simcse_path", type=str, help="Path of simcse model")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    simcse = SimCSE(args.simcse_path)
    warnings_inference_file = json.load(open(args.warnings_inference_file, "r", encoding="utf-8"))
    warnings_file = json.load(open(args.warnings_file, "r", encoding="utf-8"))
    wikihow_file = json.load(open(args.wikihow_file, "r", encoding="utf-8"))

    keys_to_delete = []
    for key in tqdm(list(warnings_inference_file.keys())):
        try:
            category_list = [article_dict["category"] for article_dict in wikihow_file if (article_dict["task"] == warnings_inference_file[key]["goal"])][0]
            category = category_filter(category_list)

            if category:
                source_tutorial = source_tutorial_filter(warnings_inference_file[key]["positive_candidate"], warnings_inference_file[key]["negative_candidates"], warnings_file)
                
                if source_tutorial:
                    length = length_filter(warnings_inference_file[key]["positive_candidate"], tokenizer, 128, 8)

                    if length:
                        similarity_for_pos_neg_candidates = similarity_filter_for_pos_neg_candidates(warnings_inference_file[key]["negative_candidates"], warnings_inference_file[key]["positive_candidate"], simcse, 0.9)

                        if similarity_for_pos_neg_candidates:
                            lexical_overlap = lexical_overlap_filter(warnings_inference_file[key]["positive_candidate"], warnings_inference_file[key]["negative_candidates"], 0.5)

                            if lexical_overlap:
                                golden_t_or_ws = warnings_file[warnings_inference_file[key]["goal"]]
                                similarity = similarity_filter(warnings_inference_file[key]["negative_candidates"], golden_t_or_ws, simcse, 0.9)

                                if similarity:
                                    all_steps = [article_dict["caption"] for article_dict in wikihow_file if (article_dict["task"] == warnings_inference_file[key]["goal"])]
                                    tutorial = " ".join([step for method in all_steps for step in method])
                                    tf_idf = tf_idf_filter(warnings_inference_file[key]["positive_candidate"], tutorial, 0.15)

                                    if tf_idf:
                                        pass

                                    else:
                                        keys_to_delete.append(key)
                                else:
                                    keys_to_delete.append(key)
                            else:
                                keys_to_delete.append(key)
                        else:
                            keys_to_delete.append(key)
                    else:
                        keys_to_delete.append(key)
                else:
                    keys_to_delete.append(key)
            else:
                keys_to_delete.append(key)
            torch.cuda.empty_cache()
        except:
            keys_to_delete.append(key)
            
    for key in keys_to_delete:
        del warnings_inference_file[key]

    for key in tqdm(list(warnings_inference_file.keys())):
        if random.randint(0, 100) < 15:
            random_index_to_change = random.randint(0, 2)
            new_positive_candidate = warnings_inference_file[key]["negative_candidates"][random_index_to_change]
            old_positive_candidate = warnings_inference_file[key]["positive_candidate"]
            new_goal = [k for k, v in warnings_file.items() if new_positive_candidate in v][0]
                
            warnings_inference_file[key]["positive_candidate"] = new_positive_candidate 
            warnings_inference_file[key]["negative_candidates"][random_index_to_change] = old_positive_candidate
            warnings_inference_file[key]["goal"] = new_goal
    
    keys_to_delete = []
    for key in tqdm(list(warnings_inference_file.keys())):
        if len(warnings_inference_file[key]["negative_candidates"]) != 3:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del warnings_inference_file[key]
    
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(warnings_inference_file, f, indent=3, ensure_ascii=False)

if __name__ == "__main__":
    main()