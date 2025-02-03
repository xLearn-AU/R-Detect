import math
import random
import datasets
import tqdm
import pandas as pd
from MGTBenchold.methods.utils import timeit
from ast import literal_eval
import os
# import sys
# import os
# sys.path.append(os.path.abspath('MGTBench/datasets'))
# you can add more datasets here and write your own dataset parsing function
DATASETS = ['meta_HC3', 'HC3', 'TruthfulQA', 'SQuAD1', 'SQuAD2',
            'NarrativeQA', "TruthfulQA_adv1", "TruthfulQA_adv2", "TruthfulQA_LLMs", "SQuAD1_LLMs","CCNews","RAID_LLMs", 'Beemo_LLMs', 'DetectRL_LLMs']

dataset_path_dit={
    'HC3': 'MGTBenchold/dataset/H3C'
}

import re
def process_spaces(text):
    text = text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    text = text.replace('\r\n', '\n').replace('\\n', '').replace('!\n', '')
    return re.sub('\n+', '\n', text)

def process_text_truthfulqa_adv(text):
    if not type(text) == str:
        return ''

    if "I am sorry" in text or "I'm sorry" in text:
        try:
            first_period = text.index('.')
        except ValueError:
            try:
                first_period = text.index(',')
            except ValueError:
                first_period = -2
        start_idx = first_period + 2
        text = text[start_idx:]
    # if "as an AI language model" in text or "As an AI language model" in text:
    if "as an AI language model" in text or "As an AI language model" in text or "I'm an AI language model" in text or "As a language model" in text:
        try:
            first_period = text.index('.')
        except ValueError:
            first_period = text.index(',')
        start_idx = first_period + 2
        text = text[start_idx:]
    ## remove any unprintable characters
    text = ''.join(x for x in text if x.isprintable())
    return text

## Use global variable to store the filtered data to avoid loading the dataset multiple times
filtered_d_HC3 = None
@timeit#Automatically calculate the running time of a function
def load_HC3(cache_dir, train_ratio=0.8):
    global filtered_d_HC3

    d = datasets.load_dataset('Hello-SimpleAI/HC3',
                              name='all', cache_dir=cache_dir)

    d = d['train']

    if filtered_d_HC3 is None:
        filtered_d_HC3 = [_ for _ in d if (len(_['human_answers']) > 0 and len(_['chatgpt_answers']) > 0 and len(_['human_answers'][0].split()) > 5 and len(_['chatgpt_answers'][0].split()) > 5)]
        filtered_d = filtered_d_HC3
    else:
        filtered_d = filtered_d_HC3

    # filtered_d = [_ for _ in d if (len(_['human_answers']) > 0 and len(_['chatgpt_answers']) > 0 and len(_['human_answers'][0].split()) > 5 and len(_['chatgpt_answers'][0].split()) > 5)]

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    # random.seed(0)
    random.shuffle(filtered_d)

    total_num = len(filtered_d)
    # total_num = 100
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * train_ratio:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(filtered_d[i]["human_answers"][0]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(filtered_d[i]["chatgpt_answers"][0]))
        data_new[data_partition]['label'].append(1)
    return data_new

@timeit
def load_meta_HC3(cache_dir):
    d = datasets.load_dataset('Hello-SimpleAI/HC3',
                              name='all', cache_dir=cache_dir)
    d = d['train']
    # filtered_d = [_ for _ in d if (len(_['human_answers']) > 0 and len(_['chatgpt_answers']) > 0 and len(_['human_answers'][0].split()) > 5 and len(
    # 	_['chatgpt_answers'][0].split()) > 5 and len(_['human_answers'][0].split()) < 150 and len(_['chatgpt_answers'][0].split()) < 150)]
    # filtered_d = [_ for _ in d if (len(_['human_answers']) > 0 and len(_['chatgpt_answers']) > 0 and len(_['human_answers'][0].split()) > 100 and len(
    # _['chatgpt_answers'][0].split()) > 100 and len(_['human_answers'][0].split()) < 180 and len(_['chatgpt_answers'][0].split()) < 180)]

    meta_data = {}
    filtered_d = [_ for _ in d if (len(_['human_answers']) > 0 and len(_['chatgpt_answers']) > 0 and len(_['human_answers'][0].split()) > 5 and len(_['chatgpt_answers'][0].split()) > 5)]

    for source in sorted(set([item['source'] for item in filtered_d])):
        meta_data[source] = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            }
        }

    # random.seed(0)
    random.shuffle(filtered_d)

    total_num = len(filtered_d)
    # total_num = 100
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.80:
            data_partition = 'train'
        else:
            data_partition = 'test'
        item = filtered_d[i]
        source = item['source']
        meta_data[source][data_partition]['text'].append(
            process_spaces(filtered_d[i]["human_answers"][0]))
        meta_data[source][data_partition]['label'].append(0)
        meta_data[source][data_partition]['text'].append(
            process_spaces(filtered_d[i]["chatgpt_answers"][0]))
        meta_data[source][data_partition]['label'].append(1)
    return meta_data

def load_CCNews(cache_dir, train_ratio=0.8):
    d = datasets.load_dataset('vblagoje/cc_news', cache_dir=cache_dir)
    d = d['train']
    filtered_d = [_ for _ in d if (len(_['title']) > 0 and len(_['description']) > 0 and len(_['text'].split()) > 5)]

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    # random.seed(0)
    random.shuffle(filtered_d)

    total_num = len(filtered_d)
    # total_num = 100
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * train_ratio:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(filtered_d[i]["text"]))
        data_new[data_partition]['label'].append(0)
    return data_new

def load_TruthfulQA(cache_dir, train_ratio=0.8):
    f = pd.read_csv("MGTBenchold/datasets/TruthfulQA_chatgpt.csv")
    q = f['Question'].tolist()
    a_human = f['Best Answer'].tolist()
    a_chat = f['chatgpt_answer'].fillna("").tolist()
    a_chat = [process_text_truthfulqa_adv(_) for _ in a_chat]
    c = f['Category'].tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 5 and len(a_chat[i].split()) > 5: # It turned out to be greater than 1. It was too short and meaningless. It was changed to greater than 4.
            res.append([q[i], a_human[i], a_chat[i], c[i]])
            ## Check if the answer has period mark at the end
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    total_num = len(res)
    random.shuffle(res)  # Randomly shuffle the data to ensure that the training set and test set are randomly distributed
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * train_ratio: #Originally it was 0.8, but due to no training, it was changed to 0.01.
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)

        # data_new[data_partition]['category'].append(res[i][3])
        # data_new[data_partition]['category'].append(res[i][3])

    return data_new

## Use global variable to cache the data to avoid loading the dataset multiple times
RAID_Cache = {}
RAID_CSV_Cache = None
def load_RAID_LLMs(cache_dir, LLM_name, raid_split="train", attack='none', max_tokens=math.inf, train_ratio=0.8, max_samples=math.inf, cache=True):
    global RAID_Cache, RAID_CSV_Cache

    # If there is already data corresponding to LLM_name in the cache, the cached result will be returned directly.
    if LLM_name != 'machine' and LLM_name in RAID_Cache:
        # Randomly shuffle data
        random.shuffle(RAID_Cache[LLM_name]['train']['text'])
        random.shuffle(RAID_Cache[LLM_name]['test']['text'])
        return RAID_Cache[LLM_name]

    if RAID_CSV_Cache is None:
        # Read CSV file
        try:
            if raid_split == "train":
                f = pd.read_csv("MGTBenchold/datasets/RAID_train.csv")
            elif raid_split == "test":
                f = pd.read_csv("MGTBenchold/datasets/RAID_test.csv")
            else:
                f = pd.read_csv("MGTBenchold/datasets/RAID_extra.csv")
            if cache:
                RAID_CSV_Cache = f
        except FileNotFoundError:
            raise FileNotFoundError(f"RAID dataset RAID_{raid_split}.csv not found, please download the dataset and place it in the MGTBenchold/datasets directory.")
    else:
        f = RAID_CSV_Cache

    # Filter data for a specified model
    if LLM_name == 'train':
        selected_data = f
    else:
        if LLM_name == 'machine':
            ## Select a model from the list of models except for human
            selected_model = random.choice([_ for _ in f['model'].unique() if _ != 'human'])
            print("RAID model selected: ", selected_model)
            selected_data = f[f['model'] == selected_model]
        else:
            selected_data = f[f['model'] == LLM_name]

        if attack == 'any':
            selected_data = selected_data[selected_data['attack'] != 'none']
        if attack != 'any' and LLM_name != 'human':
            selected_data = selected_data[selected_data['attack'] == attack]

    # Extract questions, answers and categories
    q = selected_data['title'].tolist()
    a = selected_data['generation'].fillna("").tolist()
    # a = [process_text_truthfulqa_adv(_) for _ in a] # Preprocessing
    for i in tqdm.tqdm(range(len(a)), desc="preprocessing"):
        a[i] = process_text_truthfulqa_adv(a[i]) # preprocessing
    c = selected_data['model'].tolist()  # Category information

    # Generate result list
    res = []
    for i in tqdm.tqdm(range(len(q)), desc="generating res"):
        if len(a[i].split()) > 5 and len(a[i].split()) < max_tokens:  # Filter longer answers
            res.append([q[i], a[i], c[i]])
            # Make sure the answer ends with a period
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
        if len(res) >= max_samples:
            break

    # Create new data structure
    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }
    }

    # Randomly shuffle data
    random.shuffle(res)
    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        # Divide training set and test set according to proportion
        data_partition = 'train' if i < total_num * train_ratio else 'test'

        # Add questions, answers and categories
        data_new[data_partition]['text'].append(process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0 if res[i][2] == 'human' else 1)  # 0 is the human answer, 1 is the machine answer
        data_new[data_partition]['category'].append(res[i][2])

    # remove unused data
    if LLM_name == 'train':
        data_new['test']['text'] = []
        data_new['test']['label'] = []
        data_new['test']['category'] = []
    else:
        data_new['train']['text'] = []
        data_new['train']['label'] = []
        data_new['train']['category'] = []
    # Cache results into RAID_Cache
    if cache:
        RAID_Cache[LLM_name] = data_new

    return data_new

## Use global variable to cache the data to avoid loading the dataset multiple times
Beemo_Cache = {}
def load_Beemo_LLMs(cache_dir, LLM_name, attack='none', raid_split='train', max_tokens=math.inf, train_ratio=0.8, max_samples=math.inf, cache=True):
    global Beemo_Cache

    # If there is already data corresponding to LLM_name in the cache, the cached result will be returned directly.
    if LLM_name != 'machine' and LLM_name in Beemo_Cache:
        # Randomly shuffle data
        random.shuffle(Beemo_Cache[LLM_name]['train']['text'])
        random.shuffle(Beemo_Cache[LLM_name]['test']['text'])
        return Beemo_Cache[LLM_name]

    # Load dataset
    f = datasets.load_dataset('toloka/beemo', cache_dir=cache_dir, split='train')

    # Filter data for a specified model
    if LLM_name == 'train':
        selected_data = f
    else:
        if LLM_name == 'machine':
            # Randomly select a model from the dataset (excluding 'human')
            selected_model = random.choice([_ for _ in list(dict.fromkeys(f['model'])) if _ != 'human'])
            print("Beemo model selected: ", selected_model)
            selected_data = f.filter(lambda x: x['model'] == selected_model)
        else:
            selected_data = f

    # Filter data and process based on attack type
    if attack == 'none':
        a_machine = [output if output else "" for output in selected_data['model_output']]
        # Print attack model
        print(f"Beemo attack model selected: {attack}")
    else:
        # Randomly select edit prompts (only if model is not 'human')
        edit_prompt = random.choice(['P1', 'P2', 'P3']) if LLM_name != 'human' else None

        if attack == 'any':
            # If attack is 'any', attack model is randomly selected
            attack = random.choice(['llama-3.1-70b', 'gpt-4o'] if LLM_name != 'human' else ['human'])
        else:
            try:
                # Parse model and edit hints from attack parameters
                attack, edit_prompt = attack.split('_')
            except ValueError:
                pass  # If parsing fails, continue to use the default value

        # Filter machine-edited output
        if f'{attack}_edits' in selected_data.column_names and attack != 'human':
            attack_data = selected_data[f'{attack}_edits']
            attack_data = [literal_eval(_)[int(edit_prompt[1])-1] for _ in attack_data if _]
            a_machine = [
                output[edit_prompt] if edit_prompt in output and output[edit_prompt] else ""
                for output in attack_data
            ]
        else:
            raise ValueError(f"Attack model '{attack}' not found in dataset.")
        # Print attack model
        print(f"Beemo attack model selected: {attack}")
        print(f"Beemo attack model edit prompt selected: {edit_prompt}")

    # Humans always use raw output
    a_human = [output if output else "" for output in selected_data['human_output']]

    # Extract questions, answers and categories
    q = selected_data['prompt']
    for i in tqdm.tqdm(range(len(a_machine)), desc="preprocessing"):
        a_machine[i] = process_text_truthfulqa_adv(a_machine[i]) # preprocessing
    c = selected_data['model']  # Category information

    # Generate result list
    res = []
    for i in tqdm.tqdm(range(len(q)), desc="generating res"):
        if len(a_human[i].split()) > 5 and len(a_human[i].split()) < max_tokens and len(a_machine[i].split()) > 5 and len(a_machine[i].split()) < max_tokens:  # Filter longer answers
            res.append([q[i], a_human[i], a_machine[i], c[i]])
            # Make sure the answer ends with a period
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'
        if len(res) >= max_samples:
            break

    # Create new data structure
    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }
    }

    # Randomly shuffle data
    random.shuffle(res)
    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * train_ratio:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)

    # remove unused data
    if LLM_name == 'train':
        data_new['test']['text'] = []
        data_new['test']['label'] = []
        data_new['test']['category'] = []
    else:
        data_new['train']['text'] = []
        data_new['train']['label'] = []
        data_new['train']['category'] = []
    # Cache results into RAID_Cache
    if cache:
        Beemo_Cache[LLM_name] = data_new

    return data_new

## Use global variable to cache the data to avoid loading the dataset multiple times
DetectRL_Cache = {}
DetectRL_CSV_Cache = None
def load_DetectRL_LLMs(cache_dir, LLM_name, raid_split="none", attack='none', max_tokens=math.inf, train_ratio=0.8, max_samples=math.inf, cache=True):
    global DetectRL_Cache, DetectRL_CSV_Cache

    # If there is already data corresponding to LLM_name in the cache, the cached result will be returned directly.
    if LLM_name != 'machine' and LLM_name in DetectRL_Cache:
        # Randomly shuffle data
        random.shuffle(DetectRL_Cache[LLM_name]['train']['text'])
        random.shuffle(DetectRL_Cache[LLM_name]['test']['text'])
        return DetectRL_Cache[LLM_name]

    if DetectRL_CSV_Cache is None:
        # Read CSV file
        f = pd.read_csv(f"MGTBenchold/datasets/DetectRL_{attack}_attack.csv")
        if cache:
            DetectRL_CSV_Cache = f
    else:
        f = DetectRL_CSV_Cache

    # Filter data for a specified model
    if LLM_name == 'train':
        selected_data = f
    else:
        if LLM_name == 'human':
            selected_data = f[f['label'] == 'human']
        else:
            selected_data = f[f['label'] != 'human']
            if LLM_name == 'machine':
                ## Select a model from the list of models except for human
                selected_model = random.choice([_ for _ in f['llm_type'].unique() if _ != 'human'])
                print("DetectRL model selected: ", selected_model)
                selected_data = selected_data[selected_data['llm_type'] == selected_model]
            else:
                selected_data = f[f['llm_type'] == LLM_name]

    # Extract questions, answers and categories
    q = selected_data['data_type'].tolist()
    a = selected_data['text'].fillna("").tolist()
    # a = [process_text_truthfulqa_adv(_) for _ in a] # Preprocessing
    for i in tqdm.tqdm(range(len(a)), desc="preprocessing"):
        a[i] = process_text_truthfulqa_adv(a[i]) # preprocessing
    c = selected_data['label'].tolist()  # Category information

    # Generate result list
    res = []
    for i in tqdm.tqdm(range(len(q)), desc="generating res"):
        if len(a[i].split()) > 5 and len(a[i].split()) < max_tokens:  # Filter longer answers
            res.append([q[i], a[i], c[i]])
            # Make sure the answer ends with a period
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
        if len(res) >= max_samples:
            break

    # Create new data structure
    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }
    }

    # Randomly shuffle data
    random.shuffle(res)
    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        # Divide training set and test set according to proportion
        data_partition = 'train' if i < total_num * train_ratio else 'test'

        # Add questions, answers and categories
        data_new[data_partition]['text'].append(process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0 if res[i][2] == 'human' else 1)  # 0 is the human answer, 1 is the machine answer
        data_new[data_partition]['category'].append(res[i][0])

    # remove unused data
    if LLM_name == 'train':
        data_new['test']['text'] = []
        data_new['test']['label'] = []
        data_new['test']['category'] = []
    else:
        data_new['train']['text'] = []
        data_new['train']['label'] = []
        data_new['train']['category'] = []
    # Cache results into DetectRL_Cache
    if cache:
        DetectRL_Cache[LLM_name] = data_new

    return data_new

def load_TruthfulQA_LLMs(cache_dir, LLM_name, train_ratio=0.8):
    f = pd.read_csv("MGTBenchold/datasets/TruthfulQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['Best Answer'].tolist()
    a_chat = f[LLM_name + '_answer'].fillna("").tolist()
    a_chat = [process_text_truthfulqa_adv(_) for _ in a_chat]
    c = f['Category'].tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 5 and len(a_chat[i].split()) > 5: # It turned out to be greater than 1. It was too short and meaningless. It was changed to greater than 4.
            res.append([q[i], a_human[i], a_chat[i], c[i]])
            ## Check if the answer has period mark at the end
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'

    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }

    }

    random.shuffle(res)
    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * train_ratio: #Originally it was 0.8, but due to no training, it was changed to 0.01.
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)

        data_new[data_partition]['category'].append(res[i][3])
        data_new[data_partition]['category'].append(res[i][3])

    return data_new


def load_TruthfulQA_adv1(cache_dir):
    f = pd.read_csv("MGTBenchold/datasets/TruthfulQA_chatgpt_prompt1.csv")
    q = f['complete_question'].tolist()
    a_human = f['Best Answer'].tolist()
    a_chat = f['chatgpt_answer'].fillna("").tolist()
    a_chat = [process_text_truthfulqa_adv(_) for _ in a_chat]
    c = f['Category'].tolist()
    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            res.append([q[i], a_human[i], a_chat[i], c[i]])
            ## Check if the answer has period mark at the end
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'

    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }

    }

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)

        data_new[data_partition]['category'].append(res[i][3])
        data_new[data_partition]['category'].append(res[i][3])

    return data_new


def load_TruthfulQA_adv2(cache_dir):
    f = pd.read_csv("MGTBenchold/datasets/TruthfulQA_chatgpt_prompt2.csv")
    q = f['complete_question'].tolist()
    a_human = f['Best Answer'].tolist()
    a_chat = f['chatgpt_answer'].fillna("").tolist()
    a_chat = [process_text_truthfulqa_adv(_) for _ in a_chat]
    c = f['Category'].tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            res.append([q[i], a_human[i], a_chat[i], c[i]])
            ## Check if the answer has period mark at the end
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'

    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }

    }

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)

        data_new[data_partition]['category'].append(res[i][3])
        data_new[data_partition]['category'].append(res[i][3])

    return data_new


def load_SQuAD1(cache_dir, train_ratio=0.8):
    f = pd.read_csv("MGTBenchold/datasets/SQuAD1_chatgpt.csv")
    q = f['Question'].tolist()
    a_human = [eval(_)['text'][0] for _ in f['answers'].tolist()]
    a_chat = f['chatgpt_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            res.append([q[i], a_human[i], a_chat[i]])
            ## Check if the answer has period mark at the end
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * train_ratio:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)
    return data_new

def load_SQuAD1_LLMs(cache_dir, LLM_name, train_ratio=0.8):
    f = pd.read_csv("MGTBenchold/datasets/SQuAD1_LLMs.csv")
    q = f['Question'].tolist()
    a_human = [eval(_)['text'][0] for _ in f['answers'].tolist()]
    a_chat = f[LLM_name+'_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            res.append([q[i], a_human[i], a_chat[i]])
            ## Check if the answer has period mark at the end
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    random.shuffle(res)
    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * train_ratio:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)
    return data_new


def load_SQuAD2(cache_dir):
    f = pd.read_csv("MGTBenchold/datasets/SQuAD2_chatgpt.csv")

    anwsers = f['answers'].tolist()
    a_chat = f['chatgpt_answer'].tolist()
    selected_index = [i for i in range(len(anwsers)) if (
        len(eval(anwsers[i])['text']) > 0 and len(a_chat[i]) > 0)]
    q = f['Question'].tolist()
    q = [q[i] for i in selected_index]

    a_human = [eval(anwsers[i])['text'][0] for i in selected_index]

    a_chat = [a_chat[i] for i in selected_index]

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            res.append([q[i], a_human[i], a_chat[i]])
            ## Check if the answer has period mark at the end
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)
    return data_new


def load_NarrativeQA(cache_dir):
    f = pd.read_csv("MGTBenchold/datasets/NarrativeQA_chatgpt.csv")
    q = f['Question'].tolist()
    a_human = f['answers'].tolist()
    a_human = [_.split(";")[0] for _ in a_human]
    a_chat = f['chatgpt_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1 and len(a_chat[i].split()) < 150 and len(a_chat[i].split()) < 150:

            res.append([q[i], a_human[i], a_chat[i]])
            ## Check if the answer has period mark at the end
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'
            if res[-1][2][-1] != '.':
                res[-1][2] += '.'

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[i][2]))
        data_new[data_partition]['label'].append(1)
    return data_new


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
