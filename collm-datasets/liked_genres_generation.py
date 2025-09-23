import pandas as pd
import transformers
import torch

import ast
import time
import datetime

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

DATA_LIST = ['test', 'valid', 'train']

model_path = ""

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

movies_df = pd.read_csv("", sep="::", header=None, names=["movieId", "title", "genres"], engine="python", encoding='latin1')
genres_map = movies_df.set_index('title')['genres'].to_dict()

for DATA_TYPE in DATA_LIST:
    start = time.time()

    print(f"\n⌛ Processing {DATA_TYPE} data...\n")

    data_ = pd.read_pickle(f"ml-1m/{DATA_TYPE}_ood2.pkl")
    
    results = []
    
    i = 0
    for history in data_['his_title']:
        if history == ['']:
            results.append("{'liked genres': ''}")
        else:
            history_text = "User history:\n"
            for item in history:
                if item != '':
                    genres = genres_map.get(item, "")
                    history_text += f"{item}, {genres}\n"
            
            prompt_text = f"""You are required to generate user profile based on the history of user, that each movie with title, year, genre.

{history_text}
Please output the following information of user, output format: {{'liked genres': ['genre1', 'genre2', 'genre3']}}
Please output exactly 3 genres that user might like in strong descending order of confidence. Even if the evidence is limited, you must make an educated guess and provide 3 genres in each. Do not leave any field empty.
Please output only the content in the format above, and nothing else. No reasoning, no analysis. Reiterating once again!! Please only output the content after "output format: ", and do not include any other content such as introduction or acknowledgments."""
            
            messages = [
                {"role": "user", "content": prompt_text}
            ]
            
            outputs = pipeline(
                messages,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.1,
                eos_token_id=pipeline.tokenizer.eos_token_id,
                pad_token_id=pipeline.tokenizer.pad_token_id
            )
                
            results.append(outputs[0]["generated_text"][-1]['content'])
            
        i += 1
        print(f"Processed {DATA_TYPE} data {i}/{len(data_)}", end="\r")
    
    liked_genres = []
    outputs = []
    
    for r in results:
        liked_genre = 'ERROR'
        
        try:
            last_brace_index = r.rfind('{')
            result = r[last_brace_index:]
            
            if result == "{'liked genres': ''}":
                pass
            elif result.endswith("']}"):
                pass
            elif result.endswith("']"):
                result += "}"
            elif result.endswith("'"):
                result += "]}"
            else:
                result += "']}"
            
            genre_dict = ast.literal_eval(result)
            liked_genre = genre_dict['liked genres']
        except:
            print(f"Error processing result: {r}")
            
        liked_genres.append(liked_genre)
        outputs.append(r)
    
    data_['liked_genres'] = liked_genres
    data_['llm_output'] = outputs
    
    data_.to_pickle(f"ml-1m_{DATA_TYPE}_ood2_genres_gt.pkl")

    print(f"\n\n✅ {DATA_TYPE} data processing completed.")

    end = time.time()
    inference_time = str(datetime.timedelta(seconds=end-start)).split(".")

    print(f"⏱️ {DATA_TYPE} data inference time {inference_time[0]}")

print("\n🚩 All data processing completed.")