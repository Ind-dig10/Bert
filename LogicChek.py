import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
from bert import tokenization

folder = 'multi_cased_L-12_H-768_A-12'

config_path = folder+'/bert_config.json'
checkpoint_path = folder+'/bert_model.ckpt'
vocab_path = folder+'/vocab.txt'

sentence_1 = "Сегодня я пошел на работу."
sentence_2 = "Жизнь - прежде всего творчество, но это не значит, что каждый человек, чтобы жить, должен родиться художником, балериной или ученым."

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)

tokens_sen_1 = tokenizer.tokenize(sentence_1)
tokens_sen_2 = tokenizer.tokenize(sentence_2)

tokens = ['[CLS]'] + tokens_sen_1 + ['[SEP]'] + tokens_sen_2 + ['[SEP]']
token_input = tokenizer.convert_tokens_to_ids(tokens)
token_input = token_input + [0] * (512 - len(token_input))

mask_input = [0] * 512

seg_input = [0]*512
len_1 = len(tokens_sen_1) + 2                   # длина первой фразы, +2 - включая начальный CLS и разделитель SEP
for i in range(len(tokens_sen_2)+1):            # +1, т.к. включая последний SEP
        seg_input[len_1 + i] = 1                # маскируем вторую фразу, включая последний SEP, единицами

# конвертируем в numpy в форму (1,) -> (1,512)
token_input = np.asarray([token_input])
mask_input = np.asarray([mask_input])
seg_input = np.asarray([seg_input])
predicts = model.predict([token_input, seg_input, mask_input])[1]
print(sentence_1, '->', sentence_2)
print('Sentence is okey:', int(round(predicts[0][0]*100)), '%')





