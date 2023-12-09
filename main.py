import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
from bert import tokenization

folder = 'multi_cased_L-12_H-768_A-12'

config_path = folder+'/bert_config.json'
checkpoint_path = folder+'/bert_model.ckpt'
vocab_path = folder+'/vocab.txt'

# необходимо переводить текст в токены
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model.summary()
sentence = 'Я купил [MASK]. И подарил ее [MASK].'
result = sentence

sentence = sentence.replace(' [MASK] ','[MASK]')
sentence = sentence.replace('[MASK] ','[MASK]')
sentence = sentence.replace(' [MASK]','[MASK]')  # удаляем лишние пробелы
sentence = sentence.split('[MASK]')             # разбиваем строку по маске
tokens = ['[CLS]']                              # фраза всегда должна начинаться на [CLS]

# обычные строки преобразуем в токены с помощью tokenizer.tokenize(), вставляя между ними [MASK]
for i in range(len(sentence)):
    vg = sentence[i]
    ft = tokenizer.tokenize('я')
    if i == 0:
        tokens = tokens + tokenizer.tokenize(sentence[i])
    else:
        tokens = tokens + ['[MASK]'] + tokenizer.tokenize(sentence[i])
tokens = tokens + ['[SEP]']

token_input = tokenizer.convert_tokens_to_ids(tokens)

token_input = token_input + [0] * (512 - len(token_input))


mask_input = [0]*512
for i in range(len(mask_input)):
    if token_input[i] == 103:
        mask_input[i] = 1

seg_input = [0]*512

token_input = np.asarray([token_input])
mask_input = np.asarray([mask_input])
seg_input = np.asarray([seg_input])

predicts = model.predict([token_input, seg_input, mask_input])[0]
predicts = np.argmax(predicts, axis=-1)
predicts = predicts[0][:len(tokens)]

out = []
# добавляем в out только слова в позиции [MASK], которые маскированы цифрой 1 в mask_input
for i in range(len(mask_input[0])):
    if mask_input[0][i] == 1:                       # [0][i], т.к. сеть возвращает batch с формой (1,512), где в первом элементе наш результат
        out.append(predicts[i])

out = tokenizer.convert_ids_to_tokens(out)          # индексы в текстовые токены

for word in out:
    result = result.replace('[MASK]', word, 1)

out = ' '.join(out)                                 # объединяем токены в строку с пробелами
out = tokenization.printable_text(out)              # в удобочитаемый текст
out = out.replace(' ##','')
print('Result:', out)
print(result)