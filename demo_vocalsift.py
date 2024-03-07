from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
import music_tag
from pprint import pprint
import json
from datetime import datetime

encoder = VoiceEncoder()


# path = Path("audio_data", "vocalsift_fma_test")
path = Path("audio_data", "fma_small_out_1")

wav_fpaths = list(path.glob("**/*.wav"))

# embeddings_dict = {
#     0: {
#         "filename": "",
#         "mehr_metadaten": ""
#     }
# }
embeddings_dict = {}

counter = len(embeddings_dict)

for file in tqdm(wav_fpaths):
    metadata = music_tag.load_file(file)
    metadata_dict = {}
    for possible_tag in metadata.tag_map:
        try:
            # print(possible_tag + ": ", end="")
            # print(metadata[possible_tag])
            if possible_tag.startswith("#"):
                # Tags die mit "#" starten sind bitrate, codec, length, channels, und samplerate - keine expliziten metadaten die wir weiterführen müssen
                pass
            else:
                metadata_dict[str(possible_tag)] = metadata[possible_tag]
                # print("metadata.tag_map[possible_tag]", metadata.tag_map[possible_tag])
                # print("metadata.tag_map[possible_tag] -> ", metadata.tag_map[possible_tag][3])
                # print("metadata.tag_map[possible_tag] -> ", "str" if "str" in str(metadata.tag_map[possible_tag][3]) else "int")
                metadata_dict[possible_tag+"_type"] = "str" if "str" in str(metadata.tag_map[possible_tag][3]) else "int"
        except:
            # print("problem with metadata, with year specifically, need to solve later")
            if possible_tag == "year":
                if "TXXX:TDAT" in metadata.mfile:
                    # print("year is supposed to be")
                    # print(metadata.mfile["TXXX:TDAT"])
                    # print(type(metadata.mfile["TXXX:TDAT"]))
                    year_string = str(metadata.mfile["TXXX:TDAT"])
                    # year_date = dateutil.parser.parse(year_string).date()
                    year = str(dateutil.parser.parse(year_string).year)
                    metadata_dict["year"] = year
                    metadata_dict["year_type"] = "str"
                    # metadata_dict["date"] = year
                    # metadata_dict["date_type"] = "str"#
                    ## "date" kennt die library nicht https://github.com/KristoforMaynard/music-tag/issues/35
            pass
    
    metadata_formatted = {}
    metadata_formatted["totaltracks"] = None # totaltracks war irgendwie immer default auf 0 (wegen "type normalization" vielleicht, siehe README) - so wird totaltracks richtig gesetzt und bleibt sonst leer, wie gewünscht
    for tag in metadata_dict:
        if not tag[::-1].startswith("epyt_"):
            # print("tag",tag)
            # print("metadata_dict[tag]", metadata_dict[tag])
            # f[tag] = "" if metadata_dict[tag+"_type"] == "str" else 0
            # f.remove_tag(tag)
            try:
                metadata_formatted.append_tag(tag, metadata_dict[tag])
            except:
                try:
                    # print(metadata_dict[tag])
                    # print(metadata_dict[tag])
                    metadata_formatted[tag] = str(metadata_dict[tag]) if metadata_dict[tag+"_type"] == "str" else int(metadata_dict[tag])
                except:
                    print("problem with tag", tag)
                    pass
    
    preprocessed_wave = preprocess_wav(wav_fpaths[0])
    # print(preprocessed_wave)

    embedding = encoder.embed_utterance(preprocessed_wave)
    # print(embedding)

    embeddings_dict[counter] = {
        "filename": str(file),
        "embedding": embedding.tolist(),        
    }
    for tag in metadata_formatted:
        embeddings_dict[counter][tag] = metadata_formatted[tag]
    counter = counter + 1

# print(Path.cwd())  
# print(Path(Path(__file__).resolve().parent.parent,"_Output"))
out_path = Path(Path(__file__).resolve().parent,"_Output")
# pprint(embeddings_dict)
json.dump( embeddings_dict, open( Path(out_path,"embeddings_"+str(datetime.now()).replace(":","-").replace(" ","_")+".json"), 'w' ),  indent = 2 )
