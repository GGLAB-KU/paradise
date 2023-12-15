import argparse 
import torch
import json
from filters import tf_idf_filter, category_filter, lexical_overlap_filter, length_filter, similarity_filter_reverse, similarity_filter_for_pos_neg_candidates
from transformers import AutoTokenizer
from tqdm import tqdm
from simcse import SimCSE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path of tokenizer")
    parser.add_argument("--reverse_warnings_inference_file", type=str, help="Path of warnings data")
    parser.add_argument("--warnings_file", type=str, help="Path of warnings data")
    parser.add_argument("--wikihow_file", type=str, help="Path of warnings data")
    parser.add_argument("--simcse_path", type=str, help="Path of simcse model")
    parser.add_argument('--save_path', type=str, help="Path of output data")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    simcse = SimCSE(args.simcse_path)
    reverse_warnings_inference_file = json.load(open(args.reverse_warnings_inference_file, "r", encoding="utf-8"))
    warnings_file = json.load(open(args.warnings_file, "r", encoding="utf-8"))
    wikihow_file = json.load(open(args.wikihow_file, "r", encoding="utf-8"))

    keys_to_delete = []
    for key in tqdm(list(reverse_warnings_inference_file.keys())):
        try:
            category_list = [article_dict["category"] for article_dict in wikihow_file if (article_dict["task"] == reverse_warnings_inference_file[key]["positive_candidate"])][0]
            category = category_filter(category_list)

            if category:
                length = length_filter(reverse_warnings_inference_file[key]["positive_candidate"], tokenizer, 32, 4)

                if length:
                    lexical_overlap = lexical_overlap_filter(reverse_warnings_inference_file[key]["positive_candidate"], reverse_warnings_inference_file[key]["negative_candidates"], 0.5)

                    if lexical_overlap:
                        goal_similarity = similarity_filter_for_pos_neg_candidates(reverse_warnings_inference_file[key]["negative_candidates"], reverse_warnings_inference_file[key]["positive_candidate"], simcse, 0.9)
                        
                        if goal_similarity:
                            golden_warnings = warnings_file[reverse_warnings_inference_file[key]["positive_candidate"]]
                            similarity = similarity_filter_reverse(reverse_warnings_inference_file[key]["negative_candidates"], golden_warnings, warnings_file, simcse, 0.9)

                            if similarity:
                                all_steps = [article_dict["caption"] for article_dict in wikihow_file if (article_dict["task"] == reverse_warnings_inference_file[key]["positive_candidate"])]
                                tutorial = " ".join([step for method in all_steps for step in method])
                                
                                for warning in reverse_warnings_inference_file[key]["warnings"]:
                                    tf_idf = tf_idf_filter(warning, tutorial, 0.15)
                                    if not tf_idf:
                                        reverse_warnings_inference_file[key]["warnings"].remove(warning)

                                if len(reverse_warnings_inference_file[key]["warnings"]) > 0:
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

            torch.cuda.empty_cache()
        except:
            keys_to_delete.append(key)
            
    for key in keys_to_delete:
        del reverse_warnings_inference_file[key]
    
    keys_to_delete = []
    for key in tqdm(list(reverse_warnings_inference_file.keys())):
        if len(reverse_warnings_inference_file[key]["negative_candidates"]) != 3:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del reverse_warnings_inference_file[key]

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(reverse_warnings_inference_file, f, indent=3, ensure_ascii=False)

if __name__ == "__main__":
    main()