import io
import os
import re
from typing import Union, Dict

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import CriolloDataset


class Criollo:
    def __init__(self, file: Union[str, io.TextIOBase, io.FileIO]) -> None:
        """

        :param file: file or path or str
        """
        if isinstance(file, str):
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    self.data = f.read() + '\n'
        elif isinstance(file, io.TextIOBase):
            self.data = file.read() + '\n'
        elif isinstance(file, io.IOBase):
            data = bytes(file.read())
            self.data = bytes.decode(data, encoding='utf-8')
        else:
            raise ValueError("Invalid argument passed")

        self.arr = None
        self.room_name = None
        self.parsed = False
        self.__parse()
        self.ai_initialized = False
        self.model = None
        self.tok = None
        self.clf = None
        self.dataloader = None

    def __parse(self):
        self.room_name = self.data[:self.data.find('님과 카카오톡 대화')].split(" ")[0]
        parser = re.compile(
            r'\[(\S+)] \[(\S{2} \d{1,2}:\d{1,2})] ([\s\S]+?)\n', re.MULTILINE)
        arr = []
        for line in parser.finditer(self.data):
            user, time, text = line.groups()
            hour, minute = map(int, time.split(" ")[1].split(":"))
            if time[1] == '후':
                hour += 12
            time = f"{hour}:{minute}"
            text = text.rstrip('\r')
            arr.append([user, time, text])
        self.arr = arr
        self.parsed = True

    def __is_valid(self, text):
        ret = True
        card = set(text)
        if len(text) > 10:
            ret = False
        if len(card) == 1:
            ret = False
        if text == '사진' or text == '이모티콘' or text == '동영상':
            ret = False
        return ret

    def count_user(self) -> Dict[str, int]:
        """

        :return: dictionary with count of text occurred by users as key
        """
        assert self.parsed
        ret = dict()
        for line in self.arr:
            user = line[0]
            if user not in ret:
                ret[user] = 0
            ret[user] += 1
        return ret

    def count_time(self, trim: str = 'hour') -> Dict[str, int]:
        """

        :param trim: trim time via given argument
        :return: dictionary with count of text occurred within time as key
        """
        assert self.parsed
        if trim == 'hour':
            def trimmer(x: str):
                return f"{x.split(':')[0]:0>2}"
        elif trim == "minute":
            def trimmer(x: str):
                return f"{x:0>2}"
        else:
            raise ValueError("Invalid argument passed")

        ret = dict()
        for line in self.arr:
            time = trimmer(line[1])
            if time not in ret:
                ret[time] = 0
            ret[time] += 1
        return ret

    def count_time_per_user(self, trim: str='hour') -> Dict[str, Dict[str, int]]:
        """
        returns count of text occurred within time for each user

        Args:
            trim (str, optional): user trim as a bin. Defaults to 'hour'.

        Returns:
            Dict[str, Dict[str, int]]: dictionary with count of text occurred within time as key for each user
        """
        assert self.parsed
        if trim == 'hour':
            def trimmer(x: str):
                return f"{x.split(':')[0]:0>2}"
        elif trim == "minute":
            def trimmer(x: str):
                return f"{x:0>2}"
        else:
            raise ValueError("Invalid argument passed")
        ret = dict()
        for line in self.arr:
            user, time, text = line
            if user not in ret:
                ret[user] = dict()
            time = trimmer(line[1])
            if time not in ret[user]:
                ret[user][time] = 0
            ret[user][time] += 1
        return ret


    def count_text(self, k=10) -> Dict[str, int]:
        """

        :param k: top text
        :return: dictionary with count of text occurred
        """
        assert self.parsed
        text_counts = dict()
        for line in self.arr:
            text = line[2]
            if not self.__is_valid(text):
                continue
            if text not in text_counts:
                text_counts[text] = 0
            text_counts[text] += 1
        text_counts = sorted(text_counts.items(),
                             key=lambda x: x[1], reverse=True)[:k]
        ret = dict()
        for key, value in text_counts:
            ret[key] = value
        return ret

    def count_text_per_user(self, top_k=10) -> Dict[str, Dict[str, int]]:
        ret = dict()
        for line in self.arr:
            user, text = line[0], line[2]
            if not self.__is_valid(text):
                continue
            if user not in ret:
                ret[user] = dict()
            if text not in ret[user]:
                ret[user][text] = 0
            ret[user][text] += 1
        for user in ret:
            temp = sorted(ret[user].items(),
                          key=lambda x: x[1], reverse=True)[:top_k]
            ret[user] = dict()
            for k, v in temp:
                ret[user][k] = v
        return ret

    def __initialize_ai(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        self.tok = AutoTokenizer.from_pretrained(
            "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis")
        device = 0 if torch.cuda.is_available() else -1
        self.clf_pipe = pipeline(task='text-classification', tokenizer=self.tok,
                                 model=self.model, device=device, use_fast=True, framework='pt')
        self.ai_initialized = True

    def sent_cls(self) -> Dict[str, list]:
        """
        conduct sentiment classification with NLU model.

        Returns:
            Dict[str, list]: dictionary list of user sentiment in time order per user
        """
        assert self.parsed
        if not self.ai_initialized:
            self.__initialize_ai()

        dataset = CriolloDataset(self.arr)
        dataloader = DataLoader(
            dataset=dataset, num_workers=4, shuffle=False, batch_size=64)

        result = dict()
        with tqdm(dataloader) as pbar:
            pbar.set_description("Analyzing text")
            for batch in pbar:
                users, times, texts = batch
                texts = list(texts)
                preds = self.clf_pipe(texts)
                for i in range(len(preds)):
                    pred, user = preds[i], users[i]
                    if user not in result:
                        result[user] = []

                    label, score = pred['label'], pred['score']
                    if score < 0.75:
                        continue

                    if label == 0:
                        result[user].append(-score)
                    else:
                        result[user].append(score)
        return result
