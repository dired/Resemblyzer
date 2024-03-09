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

# VocalSift
import hashlib # hash of inputfile as key of dict


out_path = Path(Path(__file__).resolve().parent,"_output")

def conglomerate_embeddings():
    print("Alle bisher gemachten embeddings werden jetzt in unique_embeddings.txt konglomeriert")
    embeddings_list = []
    for embeddings_file in list(out_path.glob("*.json")):
        embedding_list = json.load(open(embeddings_file,"r"))
        embeddings_list.append(embedding_list)

    # eigentlich kann man alles in ein dictionary anfügen, weil sha256 immer unique keys gibt
    ## das ist jetzt nur statistik fürs Nachvollziehen: Wieviele unique embeddings gibt es, wieviele embeddings kommen doppelt vor
    unique_embeddings_keys = []
    unique_embeddings = []
    duplicate_embeddings_keys = []
    for i, liste in enumerate(embeddings_list):
        print("Länge der Liste",str(i),":",len(liste))
        all_other_embeddings = [l for j, l in enumerate(embeddings_list) if j != i]
        if all_other_embeddings: # falls nicht nur 1 .json-file in _output
            for other_embedding in all_other_embeddings:
                other_keys = other_embedding.keys()
                for key in liste.keys():
                    if key in other_keys:
                        if key not in duplicate_embeddings_keys:
                            duplicate_embeddings_keys.append(key)
                    if key not in unique_embeddings:
                        unique_embeddings_keys.append(key)
                        unique_embeddings.append({key:liste[key]})
        else:
            unique_embeddings_keys.append(liste.keys())
            unique_embeddings=liste

    json.dump(unique_embeddings, open( Path(out_path,"unique_embeddings.txt"), 'w' ),  indent = 2 ) # als txt abgespeichert, damit es nicht als normales embeddings-.json erkannt wird
    print("Konglomeration beendet! unique_embeddings.txt ist jetzt aktuell")
    print("length of unique embeddings",len(unique_embeddings_keys))
    print("length of duplicate embeddings",len(duplicate_embeddings_keys))
    


def main(path):
    encoder = VoiceEncoder() # -> "loaded the voice encoder model on ..."
    wav_fpaths = list(path.glob("**/*.wav"))

    # embeddings_dict = {
    #     0: {
    #         "filename": "",
    #         "mehr_metadaten": ""
    #     }
    # }
    embeddings_dict = {}

    # counter = len(embeddings_dict) # = 0 normalerweise

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
            if not tag[::-1].startswith("epyt_"): # TODO: endswith("_type") testen, das macht genau dasselbe
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
        
        print("preprocess wav was before",wav_fpaths[0])
        print("preprocess wav is now",file)
        preprocessed_wave = preprocess_wav(file)
        # sys.exit()
        # print(preprocessed_wave)

        embedding = encoder.embed_utterance(preprocessed_wave)
        # print(embedding)

        # hash des dateinamen als key, um neue/bekannte dateien zu erkennen und aufzunehmen

        hash_object = hashlib.sha256()
        with open(file,'rb') as f:
            data = f.read()
            hash_object.update(data)

        # embeddings_key = counter  # war vorher
        embeddings_key = hash_object.hexdigest()

        embeddings_dict[embeddings_key] = {
            "filename": str(file),
            "embedding": embedding.tolist(),
            "date_embedding_generated": str(datetime.today())
        }
        for tag in metadata_formatted:
            embeddings_dict[embeddings_key][tag] = metadata_formatted[tag]
        # counter = counter + 1

    # print(Path.cwd())  
    # print(Path(Path(__file__).resolve().parent.parent,"_output"))
    
    # pprint(embeddings_dict)
    outfilename = "embeddings_"+str(datetime.now()).replace(":","-").replace(" ","_")+".json"
    json.dump( embeddings_dict, open( Path(out_path,outfilename), 'w' ),  indent = 2 )
    conglomerate_embeddings()
    return outfilename

if __name__ == "__main__":
    # path = Path("audio_data", "vocalsift_fma_test")
    # path = Path("audio_data", "fma_small_out_1")
    path = Path("..","..","..","_Cloud-HPC","Newell_fachgebiet","Output")
    main(path)