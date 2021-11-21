import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pymorphy2
from pymystem3 import Mystem as mystem
import string
from sklearn.linear_model import SGDClassifier
import re
from nltk.stem import *
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from nltk.stem.snowball import SnowballStemmer 
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
mystem = Mystem() 


st.header('Введите подкатегорию:')
podkat = st.text_area('', 'Введите подкатегорию:', height=50)
st.header('Введите наименование:')
name = st.text_area('', 'Введите наименование:', height=50)

def MainF(name, podkat):
	dt = pd.read_excel("/Реестр 327 тыс. деклараций ЕП РФ без 140000-200000.xlsx", 0)
	dt.columns = [x.replace(" ", "_") for x in dt.columns]
	def razb(txt, sp):
		for i in range(txt.count(";")):
			id = txt.index(";")
			sp.append(txt[:id])
			txt = txt[id+2:]
		sp.append(txt)
		return sp
		dop_sp = []
		delet = []
		n, m = dt.shape
		for i in range(n):
			if ";" in dt.iloc[i, 2]:
				delet.append(i)
				nu = []
				st = []
				razb(dt.iloc[i, 2], nu)
				razb(dt.iloc[i, 3], st)
				for j in range(len(nu)):
					dop_sp.append([dt.iloc[i, 0], dt.iloc[i, 1], nu[j], st[j]])
		for k in dop_sp:
  			dt.loc[len(dt.index)] = k
		for el in delet[::-1]:
			dt = dt.drop(el)


	def remove_punctuation(text):
		return "".join([ch if ch not in string.punctuation else ' ' for ch in text])

	def remove_numbers(text):
		return ''.join([i if not i.isdigit() else ' ' for i in text])

	def remove_multiple_spaces(text):
		return re.sub(r'\s+', ' ', text, flags=re.I)
	russian_stopwords = stopwords.words("russian")
	russian_stopwords.extend(['…', '«', '»', '...'])
	def lemmatize_text(text):
		tokens = mystem.lemmatize(text.lower())
		tokens = [token for token in tokens if token not in russian_stopwords and token != " "]
		text = " ".join(tokens)
		return text

	preproccessing = lambda text: (remove_multiple_spaces(remove_numbers(remove_punctuation(text))))
	dt_sort['preproccessed'] = list(map(preproccessing, dt_sort["Общее_наименование_продукции"]))	
	prep_text = [remove_multiple_spaces(remove_numbers(remove_punctuation(text.lower()))) for text in dt_sort["Общее_наименование_продукции"]]
	dt_sort['text_prep'] = prep_text

	#stemming
	stemmer = SnowballStemmer("russian")
	russian_stopwords = stopwords.words("russian")
	russian_stopwords.extend(['…', '«', '»', '...', 'т.д.', 'т', 'д'])
	text = dt_sort["text_prep"][0]
	stemmed_texts_list = []
	for text in dt_sort['text_prep']:
		tokens = word_tokenize(text)    
		stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in russian_stopwords]
		text = " ".join(stemmed_tokens)
		stemmed_texts_list.append(text)

	dt_sort['text_stem'] = stemmed_texts_list
	def remove_stop_words(text):
		tokens = word_tokenize(text) 
		tokens = [token for token in tokens if token not in russian_stopwords and token != ' ']
		return " ".join(tokens)
	sw_texts_list = []
	for text in dt_sort['text_prep']:
		tokens = word_tokenize(text)    
		tokens = [token for token in tokens if token not in russian_stopwords and token != ' ']
		text = " ".join(tokens)
		sw_texts_list.append(text)

	dt_sort ['text_sw'] = sw_texts_list	

	#####limitizatia
	morph = pymorphy2.MorphAnalyzer()

	def lemmatize(text):
		words = text.split() # разбиваем текст на слова
		res = list()
		for word in words:
			p = morph.parse(word)[0]
			res.append(p.normal_form)

		return res

	lem_txt = []
	for txt in dt_sort['text_sw']:
		x = lemmatize(txt)
		l = " ".join(x)  
		lem_txt.append(l)
	dt_sort['text_lem'] = lem_txt
	rez = []
	for txt in dt_sort['Подкатегория_продукции'].unique():
		if ";" in txt:
			for i in range(txt.count(";")):
				id = txt.index(";")
				rez.append(txt[:id])
				txt = txt[id+2:]
		rez.append(txt)
	my_tags = list(set(rez))

	y = my_tags
	X = dt_sort['text_lem']
	y = dt_sort['Подкатегория_продукции']
	z = podkat
	dbm = []
	bkb = table.split('\n')
	for i in range(len(bkb)- 1):
		dbm.append(bkb[i + 1].strip())
	for txt in dbm:
		fk = txt.split('       ')
		if fk[0] == z:
			if fk[1][:4] >= 0.75:
				name = (remove_multiple_spaces(remove_numbers(remove_punctuation(name))))
				name = remove_stop_words(name)
				name = lemmatize_text(name)
				pred = logreg.predict([name])
				st.write(pred)
			else:
				st.write("Don't know")
			print(float(fk[1][:4]))






send = st.button('Отправить')
if(send):
	MainF(name, podkat)
else:
	st.write('.....')


