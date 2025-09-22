import pandas as pd
import re
from transformers import AutoTokenizer
from razdel import tokenize as razdel_tokenize


def find_spaces(text: str) -> list[int]:
    '''
    Поиск индексов пробелов

    На вход подаётся строка с восстановленными пробелами
    '''
    spaces = list()
    accumed_text = 0 # накопившийся обрезанный текст для правильной индексации
    accumed_spaces = 0 # накопившиеся пробелы для приведения к слитной строке
    while True:
        space_idx = text.find(" ")
        if space_idx == -1: # условие выхода - ненаход
            break
        # логика подсказана здравым смыслом и опытом, тесты проходит
        spaces.append(accumed_text + space_idx - accumed_spaces)

        text = text[space_idx+1:] # +1 для пропуска найденного пробела

        accumed_text += space_idx + 1 # см. коммент выше
        accumed_spaces += 1 # очев
    return spaces


def make_submission(df, restored: pd.Series):
    '''
    restored - pd.Series из строк с восстановленными пробелами
    '''

    df["predicted_positions"] = restored.apply(find_spaces).apply(str)
    res = df.drop(columns=("text_no_spaces"))
    res.to_csv("submission.csv", index=False)


class Segmenter:
    def __init__(self, path):
        with open(path, 'r', encoding="utf-8") as file:
            self.vocab = set(word.strip().lower() for word in file if word.strip())

        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru") 

        self._ascii_re = re.compile(r'^[\x00-\x7f]+$')
        self._punct_before_re = re.compile(r'\s+([.,!?;:])')
        self._open_after_re = re.compile(r'([“"\'«(\[{])\s+')
        self._close_before_re = re.compile(r'\s+([”"\'»)\]}])')
    
    def _is_ascii(self, token: str) -> bool:
        return bool(self._ascii_re.match(token))
        
    def _build_from_subtokens(self, token: str, subtokens: list[str]) -> list[str]:
        # склейка сабтокенов по словарю
        buffer = ""
        parts = list()
        for st in subtokens:
            piece = st.lstrip("#")
            if not piece: 
                continue
            buffer += piece
            if buffer.lower() in self.vocab:
                parts.append(buffer)
                buffer = ""
        if buffer == "":
            return parts # успешно разобран
        
        remained = buffer
        tmp = list()
        while remained:
            matched = False
            for l in range(len(remained), 0, -1):
                pref = remained[:l]
                if pref.lower() in self.vocab:
                    tmp.append(pref)
                    remained = remained[l:]
                    matched = True
                    break
            if not matched:
                return None
        parts.extend(tmp)
        return parts
    
    def preprocessor(self, text: str) -> str:
        # базовая сегментация razdel
        tokens = [t.text for t in razdel_tokenize(text)]
        text = " ".join(tokens)

        # удаление лишних пробелов перед знаками препинания и вокруг кавычек
        text = self._punct_before_re.sub(r"\1", text)
        text = self._open_after_re.sub(r"\1", text)
        text = self._close_before_re.sub(r"\1", text)

        return text

    def segment(self, text: str) -> str:
        tokens = text.split()
        out_tokens = list()

        for token in tokens:
            # англ / числа / пунктуация
            if self._is_ascii(token):
                out_tokens.append(token)
                continue
            
            # токенизация 
            subtokens = self.tokenizer.tokenize(token)
            parts = self._build_from_subtokens(token, subtokens) # попытка склейки слов по словарю
            if parts is not None:
                out_tokens.extend(parts)
                continue
            

            # восстановление строки при частично удачной или неудачной попытке склейки
            remained = token
            tmp = list()
            while remained:
                matched = False
                for l in range(len(remained), 0, -1):
                    pref = remained[:l]
                    if pref.lower() in self.vocab:
                        tmp.append(pref)
                        remained = remained[l:]
                        matched = True
                        break
                if not matched:
                    # сохраняем только остаток, а не весь токен
                    tmp.append(remained)
                    remained = ""
            out_tokens.extend(tmp)

        # финальная чистка
        s = " ".join(out_tokens)
        s = self._punct_before_re.sub(r"\1", s)
        s = self._open_after_re.sub(r"\1", s)
        s = self._close_before_re.sub(r"\1", s)

        return s
    
    def process(self, text: str) -> str:
        return self.segment(self.preprocessor(text))


# read_csv ломается, так как в строках переменное число запятых
rows = list()
with open("dataset_1937770_3.txt", encoding="utf-8", mode='r') as file:
    header = next(file).rstrip('\n').split(',') 
    for line in file:
        line = line.rstrip('\n')
        if not line:
            continue
        id_str, text_no_space = line.split(',', maxsplit=1) # только первая запятая
        rows.append((int(id_str), text_no_space))

df = pd.DataFrame(data=rows, columns=header)

model = Segmenter(path="words_cases.txt")

# для визуальной оценки
# for text in df.text_no_spaces:
#     restored = model.process(text)
#     print(restored)

restored = df['text_no_spaces'].apply(model.process)
make_submission(df, restored)

# Mean F1 = 59.659%