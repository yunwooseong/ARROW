# 에폭마다 빈 sequence에서 random sampling 넣어서 해보기

import os
from select import select
# from PIL import Image
# import webdataset as wds
from minigpt4.datasets.datasets.rec_base_dataset import RecBaseDataset 
import pandas as pd
import numpy as np
import logging
import torch # MODIFIED
import random # MODIFIED
# from minigpt4.datasets.datasets.caption_datasets import CaptionDataset




# class RecDataset(RecBaseDataset):


#     def __getitem__(self, index):

#         # TODO this assumes image input, not general enough
#         ann = self.annotation.iloc[index]
#         return {
#             "User": ann['User'],
#             "InteractedItems": ann['InteractedItems'],
#             "InteractedItemTitles": ann['InteractedItemTitles'],
#             "TargetItemID": ann["TargetItemID"],
#             "TargetItemTitle": ann["TargetItemTitle"]
#         }
        

def convert_title_list_v2(titles):
    titles_ = []
    for x in titles:
        if len(x)>0:
            titles_.append("\""+ x + "\"")
    if len(titles_)>0:
        return ", ".join(titles_)
    else:
        return "unkow"
def convert_title_list(titles):
    titles = ["\""+ x + "\"" for x in titles]
    return ", ".join(titles)
    

class MovielensDataset(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(text_processor, ann_paths)
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+".pkl").reset_index(drop=True)
        self.use_his = False
        if 'sessionItems' in self.annotation.columns:
            self.use_his = True
            self.annotation = self.annotation[['uid','iid','title','sessionItems', 'sessionItemTitles','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(convert_title_list)
        else:
            self.annotation = self.annotation[['uid','iid','title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle','label']
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length = max(max_length,len(x))
            self.max_lenght = max_length
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            else:
                b = a
            return {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": ann['InteractedItemTitles'],
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            return {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                # "InteractedNum": None,
                "label": ann['label']
            }



class MovielensDataset_stage1(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(text_processor, ann_paths)
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+".pkl").reset_index(drop=True)[['uid','iid','title','sessionItems', 'sessionItemTitles','label', 'pairItems', 'pairItemTitles']]
        self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label','PairItemIDs','PairItemTitles']
        self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(convert_title_list)
        self.annotation["PairItemTitles"] = self.annotation["PairItemTitles"].map(convert_title_list)
        
        
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        return {
            "UserID": ann['UserID'],
            "PairItemIDs": np.array(ann['PairItemIDs']),
            "PairItemTitles": ann["PairItemTitles"],
            "label": ann['label']
        }

class AmazonDataset(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+"_seqs.pkl").reset_index(drop=True)
        self.use_his = False
        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            self.use_his = True
            self.annotation = self.annotation[['uid','iid','title','his', 'his_title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            self.annotation = self.annotation[['uid','iid','title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle','label']
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_,len(x))
            self.max_lenght = min(max_length_, 15) # average: only 5 
            print("amazon datasets, max history length:", self.max_lenght)
            logging.info("amazon datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            return {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            return {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"],
                # "InteractedNum": None,
                "label": ann['label']
            }


class MoiveOOData(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        ann_paths = ann_paths[0].split("=")
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2_genres_normalized.pkl").reset_index(drop=True) # _genres_normalized

        ## warm test:
        if "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['warm'].isin([1])].copy()
        if "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()
        
        
        

        self.use_his = False
        self.prompt_flag = False

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','title','his', 'his_title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
            
            if 'liked_genres' in self.annotation.columns:
                used_columns.append('liked_genres')
                renamed_columns.append('LikedGenres')
            
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
            
            if 'liked_genres' in self.annotation.columns:
                used_columns.append('liked_genres')
                renamed_columns.append('LikedGenres')
            
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor

        ### ===== MODIFIED ===== ###
        GENRE_PRIORITY = {
            'drama': 0, 'comedy': 1, 'thriller': 2, 'crime': 3, 'action': 4,
            'science fiction': 5, 'adventure': 6, 'classic': 7, 'romance': 8, 'horror': 9,
            'animation': 10, 'biography': 11, 'musical': 12, 'war': 13, 'fantasy': 14,
            'historical': 15, 'mystery': 16, 'western': 17, 'sports': 18, 'documentary': 19,
            'independent': 20, 'satire': 21,
        }
        self.genres_list = sorted(GENRE_PRIORITY)
        self.genre_to_idx_map = {genre: i for i, genre in enumerate(self.genres_list)}
        ### ===== MODIFIED (END) ===== ###
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10) # average: only 50; 0915: 15 
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        one_sample = {}

        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }

        if self.prompt_flag:
            one_sample['prompt_flag'] = ann['prompt_flag']

        ### ===== MODIFIED ===== ###
        for genre_type in ['LikedGenres']:
            key_indices = f'{genre_type.lower()}_indices'
            key_original = genre_type

            if key_original in ann and isinstance(ann[key_original], list):
                indices = [self.genre_to_idx_map.get(str(g).lower(), -1) for g in ann[key_original]]
                indices = [idx for idx in indices if idx != -1]

                padded_indices = indices[:3] + [-100] * (3 - len(indices))
                one_sample[key_indices] = torch.tensor(padded_indices, dtype=torch.long)
            else:
                one_sample[key_indices] = torch.tensor([-100, -100, -100], dtype=torch.long)
        ### ===== MODIFIED (END) ===== ###

        return one_sample


class MoiveOOData_sasrec(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None,sas_seq_len=25):
        super().__init__()
        ann_paths = ann_paths[0].split("=")
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2_genres_normalized.pkl").reset_index(drop=True)
        
        self.use_his = False
        self.prompt_flag = False
        self.sas_seq_len = sas_seq_len

        if "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['warm'].isin([1])].copy()
        if "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','title','his', 'his_title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
            
            if 'liked_genres' in self.annotation.columns:
                used_columns.append('liked_genres')
                renamed_columns.append('LikedGenres')

            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"]
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']

            if 'liked_genres' in self.annotation.columns:
                used_columns.append('liked_genres')
                renamed_columns.append('LikedGenres')

            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        GENRE_PRIORITY = {
            'drama': 0, 'comedy': 1, 'thriller': 2, 'crime': 3, 'action': 4,
            'science fiction': 5, 'adventure': 6, 'classic': 7, 'romance': 8, 'horror': 9,
            'animation': 10, 'biography': 11, 'musical': 12, 'war': 13, 'fantasy': 14,
            'historical': 15, 'mystery': 16, 'western': 17, 'sports': 18, 'documentary': 19,
            'independent': 20, 'satire': 21,
        }
        self.genres_list = sorted(GENRE_PRIORITY)
        self.genre_to_idx_map = {genre: i for i, genre in enumerate(self.genres_list)}

        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10)
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        one_sample = {}

        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a))
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            
            if len(a) < self.sas_seq_len:
                c = [0]*(self.sas_seq_len - len(a))
                c.extend(a)
            elif len(a) >= self.sas_seq_len:
                c = a[-self.sas_seq_len:]

            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label'],
                "sas_seq": np.array(c)
            }
        else:
            one_sample = {
                "UserID": ann['UserID'],
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                "label": ann['label']
            }

        if self.prompt_flag:
            one_sample['prompt_flag'] = ann['prompt_flag']
        
        for genre_type in ['LikedGenres']:
            key_indices = f'{genre_type.lower()}_indices'
            key_original = genre_type

            if key_original in ann and isinstance(ann[key_original], list):
                indices = [self.genre_to_idx_map.get(str(g).lower(), -1) for g in ann[key_original]]
                indices = [idx for idx in indices if idx != -1]

                padded_indices = indices[:3] + [-100] * (3 - len(indices))
                one_sample[key_indices] = torch.tensor(padded_indices, dtype=torch.long)
            else:
                one_sample[key_indices] = torch.tensor([-100, -100, -100], dtype=torch.long)

        return one_sample


class AmazonOOData(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        ann_paths = ann_paths[0].split('=') 
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2_genres_normalized.pkl").reset_index(drop=True)
        self.use_his = False
        self.prompt_flag = False

        # ## warm test:
        
        if 'not_cold' in self.annotation.columns and "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([1])].copy()
        if 'not_cold' in self.annotation.columns and "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','title','his', 'his_title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']

            if 'liked_genres' in self.annotation.columns:
                used_columns.append('liked_genres')
                renamed_columns.append('LikedGenres')

            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']

            if 'liked_genres' in self.annotation.columns:
                used_columns.append('liked_genres')
                renamed_columns.append('LikedGenres')

            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        GENRE_PRIORITY = {
            'romance': 0, 'mystery': 1, 'contemporary': 2, 'fantasy': 3, 'thriller': 4,
            'historical': 5, 'crime': 6, 'paranormal': 7, 'adult': 8, 'sci-fi': 9,
            'suspense': 10, 'erotica': 11, 'billionaire': 12, 'historical': 13, 'sports': 14,
            'western': 15, 'adventure': 16, 'memoir': 17, 'biography': 18, 'christian': 19,
            'self-help': 20, 'military': 21
        }
        self.genres_list = sorted(GENRE_PRIORITY)
        self.genre_to_idx_map = {genre: i for i, genre in enumerate(self.genres_list)}
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10) # average: only 50; 0915: 15 
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        one_sample = {}
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
        
        if self.prompt_flag:
            one_sample['prompt_flag'] = ann['prompt_flag']
        
        for genre_type in ['LikedGenres']:
            key_indices = f'{genre_type.lower()}_indices'
            key_original = genre_type

            if key_original in ann and isinstance(ann[key_original], list):
                indices = [self.genre_to_idx_map.get(str(g).lower(), -1) for g in ann[key_original]]
                indices = [idx for idx in indices if idx != -1]

                padded_indices = indices[:3] + [-100] * (3 - len(indices))
                one_sample[key_indices] = torch.tensor(padded_indices, dtype=torch.long)
            else:
                one_sample[key_indices] = torch.tensor([-100, -100, -100], dtype=torch.long)

        return one_sample 



class AmazonOOData_sasrec(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None,sas_seq_len=20):
        super().__init__()
        ann_paths = ann_paths[0].split('=')
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2_genres_normalized.pkl").reset_index(drop=True)
        
        self.use_his = False
        self.prompt_flag = False
        self.sas_seq_len = sas_seq_len

        if 'not_cold' in self.annotation.columns and "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([1])].copy()
        if 'not_cold' in self.annotation.columns and "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','title','his', 'his_title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']

            if 'liked_genres' in self.annotation.columns:
                used_columns.append('liked_genres')
                renamed_columns.append('LikedGenres')

            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"]
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']

            if 'liked_genres' in self.annotation.columns:
                used_columns.append('liked_genres')
                renamed_columns.append('LikedGenres')

            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        GENRE_PRIORITY = {
            'romance': 0, 'mystery': 1, 'contemporary': 2, 'fantasy': 3, 'thriller': 4,
            'historical': 5, 'crime': 6, 'paranormal': 7, 'adult': 8, 'sci-fi': 9,
            'suspense': 10, 'erotica': 11, 'billionaire': 12, 'historical': 13, 'sports': 14,
            'western': 15, 'adventure': 16, 'memoir': 17, 'biography': 18, 'christian': 19,
            'self-help': 20, 'military': 21
        }
        self.genres_list = sorted(GENRE_PRIORITY)
        self.genre_to_idx_map = {genre: i for i, genre in enumerate(self.genres_list)}

        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10)
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        one_sample = {}

        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a))
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            
            if len(a) < self.sas_seq_len:
                c = [0]*(self.sas_seq_len - len(a))
                c.extend(a)
            elif len(a) >= self.sas_seq_len:
                c = a[-self.sas_seq_len:]

            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label'],
                "sas_seq": np.array(c)
            }
        else:
            one_sample = {
                "UserID": ann['UserID'],
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                "label": ann['label']
            }

        if self.prompt_flag:
            one_sample['prompt_flag'] = ann['prompt_flag']

        for genre_type in ['LikedGenres']:
            key_indices = f'{genre_type.lower()}_indices'
            key_original = genre_type

            if key_original in ann and isinstance(ann[key_original], list):
                indices = [self.genre_to_idx_map.get(str(g).lower(), -1) for g in ann[key_original]]
                indices = [idx for idx in indices if idx != -1]

                padded_indices = indices[:3] + [-100] * (3 - len(indices))
                one_sample[key_indices] = torch.tensor(padded_indices, dtype=torch.long)
            else:
                one_sample[key_indices] = torch.tensor([-100, -100, -100], dtype=torch.long)

        return one_sample