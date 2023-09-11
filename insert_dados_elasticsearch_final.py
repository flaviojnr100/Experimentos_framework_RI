from haystack.document_stores import ElasticsearchDocumentStore
import pandas as pd

class Savoy:

    def __removeAllPTAccent(self, old_word):
        word = list(old_word)
        len_word = len(word)-1
        for i in range(len_word, -1, -1):
            if word[i] == 'ä':
                word[i] = 'a'
            if word[i] == 'â':
                word[i] = 'a'
            if word[i] == 'à':
                word[i] = 'a'
            if word[i] == 'á':
                word[i] = 'a'
            if word[i] == 'ã':
                word[i] = 'a'
            if word[i] == 'ê':
                word[i] = 'e'
            if word[i] == 'é':
                word[i] = 'e'
            if word[i] == 'è':
                word[i] = 'e'
            if word[i] == 'ë':
                word[i] = 'e'
            if word[i] == 'ï':
                word[i] = 'i'
            if word[i] == 'î':
                word[i] = 'i'
            if word[i] == 'ì':
                word[i] = 'i'
            if word[i] == 'í':
                word[i] = 'i'
            if word[i] == 'ü':
                word[i] = 'u'
            if word[i] == 'ú':
                word[i] = 'u'
            if word[i] == 'ù':
                word[i] = 'u'
            if word[i] == 'û':
                word[i] = 'u'
            if word[i] == 'ô':
                word[i] = 'o'
            if word[i] == 'ö':
                word[i] = 'o'
            if word[i] == 'ó':
                word[i] = 'o'
            if word[i] == 'ò':
                word[i] = 'o'
            if word[i] == 'õ':
                word[i] = 'o'
            if word[i] == 'ç':
                word[i] = 'c'

        new_word = "".join(word)
        return new_word

    def __finalVowelPortuguese(self, word):
        len_word = len(word)
        if len_word > 3:
            if word[-1] == 'e' or word[-1] == 'a' or word[-1] == 'o':
                word = word[:-1]

        return word

    def __remove_PTsuffix(self, word):
        len_word = len(word)

        if len_word > 3:
            if word[-1] == 's' and word[-2] == 'e' and (word[-3] == 'r' or word[-3] == 's' or word[-3] == 'z' or word[-3] == 'l'):
                word = word[:-2]
                return word
        if len_word > 2:
            if word[-1] == 's' and word[-2] == 'n':
                new_word = list(word)
                new_word[-2] = 'm'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 3:
            if (word[-1] == 's' and word[-2] == 'i') and (word[-3] == 'e' or word[-3] == 'é'):
                new_word = list(word)
                new_word[-3] = 'e'
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 3:
            if word[-1] == 's' and word[-2] == 'i' and word[-3] == 'a':
                new_word = list(word)
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 3:
            if word[-1] == 's' and word[-2] == 'i' and word[-3] == 'ó':
                new_word = list(word)
                new_word[-3] = 'o'
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 3:
            if word[-1] == 's' and word[-2] == 'i':
                new_word = list(word)
                new_word[-1] = 'l'
                sing = "".join(new_word)
                return sing

        if len_word > 2:
            if word[-1] == 's' and word[-2] == 'e' and word[-3] == 'õ':
                new_word = list(word)
                new_word[-3] = 'ã'
                new_word[-2] = 'o'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing
            if word[-1] == 's' and word[-2] == 'e' and word[-3] == 'ã':
                new_word = list(word)
                new_word[-2] = 'o'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 5:
            if word[-1] == 'e' and word[-2] == 't' and word[-3] == 'n' and word[-4] == 'e' and word[-5] == 'm':
                word = word[:-5]
                return word

        if len_word > 2:
            if word[-1] == 's':
                word = word[:-1]

        return word

    def __normFemininPortuguese(self, word):

        len_word = len(word)

        if len_word < 3 or word[-1] != 'a':
            return word

        if len_word > 6:

            if word[-2] == 'h' and word[-3] == 'n' and word[-4] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'c' and word[-3] == 'a' and word[-4] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'r' and word[-3] == 'i' and word[-4] == 'e':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

        if len_word > 5:
            if word[-2] == 'n' and word[-3] == 'o':
                new_word = list(word)
                new_word[-3] = 'ã'
                new_word[-2] = 'o'
                masc = "".join(new_word)
                masc = masc[:-1]
                return masc

            if word[-2] == 'r' and word[-3] == 'o':
                word = word[:-1]
                return word

            if word[-2] == 's' and word[-3] == 'o':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 's' and word[-3] == 'e':
                new_word = list(word)
                new_word[-3] = 'ê'
                masc = "".join(new_word)
                masc = masc[:-1]
                return masc

            if word[-2] == 'c' and word[-3] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'd' and word[-3] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'd' and word[-3] == 'a':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'v' and word[-3] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'm' and word[-3] == 'a':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'n':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

        return word

    def stem(self, word):
        len_word = len(word)
        if len_word > 2:
            word = self.__remove_PTsuffix(word)
            word = self.__normFemininPortuguese(word)
            word = self.__finalVowelPortuguese(word)
            word = self.__removeAllPTAccent(word)

        return word


class RSLP_S:
    def __plural_reduction(self, word):
        excep = ["lápis","cais","mais","crúcis","biquínis","pois","depois","dois","leis" ]
        excep_s = ["aliás","pires","lápis","cais","mais","mas","menos", "férias","fezes","pêsames","crúcis","gás", "atrás","moisés","através","convés","ês", "país","após","ambas","ambos","messias"]

        len_word = len(word)
        new_word = list(word)

        if len_word >= 3:
            if new_word[-1] == 's' and new_word[-2] == 'n':
                new_word[-2] = 'm'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

            if new_word[-1] == 's' and new_word[-2] == 'e' and new_word[-3] == 'õ':
                new_word[-3] = 'ã'
                new_word[-2] = 'o'
                sing = "".join(new_word)
                sing = sing[:-1]
                return  sing

            if new_word[-1] == 's' and new_word[-2] == 'e' and new_word[-3] == 'ã':
                if word == 'mães':
                    word = word[:-1]
                    return word
                else:
                    new_word[-2] = 'o'
                    sing = "".join(new_word)
                    sing = sing[:-1]
                    return sing

            if new_word[-1] == 's' and new_word[-2] == 'i' and new_word[-3] == 'a':
                if word != 'cais' and word != 'mais':
                    new_word[-2] = 'l'
                    sing = "".join(new_word)
                    sing = sing[:-1]
                    return sing

            if new_word[-1] == 's' and new_word[-2] == 'i' and new_word[-3] == 'é':
                new_word[-3] = 'e'
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

            if new_word[-1] == 's' and new_word[-2] == 'i' and new_word[-3] == 'e':
                new_word[-3] = 'e'
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

            if new_word[-1] == 's' and new_word[-2] == 'i' and new_word[-3] == 'ó':
                new_word[-3] = 'o'
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

            if new_word[-1] == 's' and new_word[-2] == 'i':
                if word not in excep:
                    new_word[-1] = 'l'
                    sing = "".join(new_word)
                    return sing

            if new_word[-1] == 's' and new_word[-2] == 'e' and new_word[-3] == 'l':
                word = word[:-2]
                return word

            if new_word[-1] == 's' and new_word[-2] == 'e' and new_word[-3] == 'r':
                word = word[:-2]
                return word

            if new_word[-1] == 's':
                if word not in excep_s:
                    word = word[:-1]

        return word

    def stem(self, word):
        word = self.__plural_reduction(word)

        return word


from nltk.tokenize import word_tokenize
from string import punctuation
import nltk
from unicodedata import normalize
from nltk.stem import RSLPStemmer
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer

def preprocess_lowercase(txt):
    txt = str(txt)
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = " ".join(terms)
    return terms

def preprocess_lowercase_pontuacao(txt):
    txt = str(txt)
    stopwords = list(punctuation)
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [word for word in terms if word not in stopwords]
    terms = " ".join(terms)
    return terms

def _remove_acentos(txt):
    txt = str(txt)
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')

def preprocess_lowercase_pontuacao_acentuacao(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [word for word in terms if word not in stopwords]
    terms = " ".join(terms)
    return terms

def preprocess_lowercase_pontuacao_acentuacao_stopword(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = nltk.corpus.stopwords.words("portuguese")
    stopwords.extend(list(punctuation))
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [word for word in terms if word not in stopwords]

    return terms

def preprocess_rslp(txt):
    txt = str(txt)
    stemmer = RSLPStemmer()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms]
    terms = " ".join(terms)
    return terms

def preprocess_rslps(txt):
    txt = str(txt)
    stemmer = RSLP_S()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms]
    terms = " ".join(terms)
    return terms

def preprocess_savoy(txt):
    txt = str(txt)
    stemmer = Savoy()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms]
    terms = " ".join(terms)
    return terms


def preprocess_lowercase_pontuacao_acentuacao_stopword_rslp(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = nltk.corpus.stopwords.words("portuguese")
    stopwords.extend(list(punctuation))

    stemmer = RSLPStemmer()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    return " ".join(terms)


def preprocess_lowercase_pontuacao_acentuacao_stopword_rslps(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = nltk.corpus.stopwords.words("portuguese")
    stopwords.extend(list(punctuation))

    stemmer = RSLP_S()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    return " ".join(terms)


def preprocess_lowercase_pontuacao_acentuacao_stopword_savoy(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = nltk.corpus.stopwords.words("portuguese")
    stopwords.extend(list(punctuation))

    stemmer = Savoy()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    return " ".join(terms)


def preprocess_bigram(txt):
    txt = str(txt)
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)


    ngram = []
    ngram_2 = list(ngrams(terms, 2))

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_trigram(txt):
    txt = str(txt)
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)


    ngram = []
    ngram_3 = list(ngrams(terms, 3))

    for w in ngram_3:
        string = w[0] + "#" + w[1]+ "#" + w[2]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_uni_bi(txt):
    txt = str(txt)
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)


    ngram = []
    ngram_1 = list(ngrams(terms, 1))
    ngram_2 = list(ngrams(terms, 2))
    for w in ngram_1:
        ngram.append(w[0])

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_bigram(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [word for word in terms if word not in stopwords]

    ngram = []
    ngram_2 = list(ngrams(terms, 2))

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_trigram(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [word for word in terms if word not in stopwords]

    ngram = []
    ngram_3 = list(ngrams(terms, 3))

    for w in ngram_3:
        string = w[0] + "#" + w[1]+ "#" + w[2]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_uni_bi(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt.lower())
    terms = [word for word in terms if word not in stopwords]

    ngram = []
    ngram_1 = list(ngrams(terms, 1))
    ngram_2 = list(ngrams(terms, 2))
    for w in ngram_1:
        ngram.append(w[0])

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)


def preprocess_lowercase_pontuacao_acentuacao_stopword_bigram_rslp(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = RSLPStemmer()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_2 = list(ngrams(terms, 2))

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_trigram_rslp(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = RSLPStemmer()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_3 = list(ngrams(terms, 3))

    for w in ngram_3:
        string = w[0] + "#" + w[1]+ "#" + w[2]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_uni_bi_rslp(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = RSLPStemmer()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_1 = list(ngrams(terms, 1))
    ngram_2 = list(ngrams(terms, 2))
    for w in ngram_1:
        ngram.append(w[0])

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_bigram_rslps(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = RSLP_S()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_2 = list(ngrams(terms, 2))

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_trigram_rslps(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = RSLP_S()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_3 = list(ngrams(terms, 3))

    for w in ngram_3:
        string = w[0] + "#" + w[1]+ "#" + w[2]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_uni_bi_rslps(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = RSLP_S()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_1 = list(ngrams(terms, 1))
    ngram_2 = list(ngrams(terms, 2))
    for w in ngram_1:
        ngram.append(w[0])

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_bigram_savoy(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = Savoy()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_2 = list(ngrams(terms, 2))

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_trigram_savoy(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = Savoy()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_3 = list(ngrams(terms, 3))

    for w in ngram_3:
        string = w[0] + "#" + w[1]+ "#" + w[2]
        ngram.append(string)

    return " ".join(ngram)

def preprocess_lowercase_pontuacao_acentuacao_stopword_uni_bi_savoy(txt):
    txt = str(txt)
    txt = _remove_acentos(txt)
    stopwords = list(punctuation)

    stemmer = Savoy()
    tokenizer = RegexpTokenizer('\w+')
    terms = tokenizer.tokenize(txt)
    terms = [stemmer.stem(word) for word in terms if word not in stopwords]

    ngram = []
    ngram_1 = list(ngrams(terms, 1))
    ngram_2 = list(ngrams(terms, 2))
    for w in ngram_1:
        ngram.append(w[0])

    for w in ngram_2:
        string = w[0] + "#" + w[1]
        ngram.append(string)

    return " ".join(ngram)



#document_store = ElasticsearchDocumentStore(host='localhost', port=9200, username='', password='', index='experimento_pre',search_fields=["content","name","pre_text_lowercase","pre_text_lowercase_pontuacao",
#                                                                                                                                                                  "pre_text_lowercase_pontuacao_acentuacao","pre_text_lowercase_pontuacao_acentuacao_stopword"
#                                                                                                                                                                  ])
#
#print("Conexao ok")
#dados = pd.read_csv(r"C:\Users\Flavio\notebooks-pesquisa-1\proposicao-tema-completo.csv", delimiter=',', encoding="utf-8")
#dados.update(dados['txtEmenta'].fillna("Nenhum"))
#dados.update(dados['imgArquivoTeorPDF'].fillna("Nenhum"))
#dados.update(dados['txtExplicacaoEmenta'].fillna("Nenhum"))
#dados.update(dados['txtIndexacao'].fillna("Nenhum"))
#print("Carregou base ok")
#dados = [
#    {
#            "content": dados.loc[i,"imgArquivoTeorPDF"],
#            "name":dados.loc[i,"txtNome"],
#            "pre_text_lowercase": preprocess_lowercase(dados.loc[i,"imgArquivoTeorPDF"]),
#            "pre_text_lowercase_pontuacao": preprocess_lowercase_pontuacao(dados.loc[i,"imgArquivoTeorPDF"]),
#            "pre_text_lowercase_pontuacao_acentuacao": preprocess_lowercase_pontuacao_acentuacao(dados.loc[i,"imgArquivoTeorPDF"]),
#            "pre_text_lowercase_pontuacao_acentuacao_stopword": preprocess_lowercase_pontuacao_acentuacao_stopword(dados.loc[i,"imgArquivoTeorPDF"]),
#
#
#    }
#   for i in range(len(dados))
#    ]
#print("Converteu")
#document_store.write_documents(dados,batch_size=9000)
#
#print("Finalizou a 1º")

#document_store = ElasticsearchDocumentStore(host='localhost', port=9200, username='', password='', index='experimento_stemming',search_fields=["content","name",
#                                                                                                                                                                  "text_savoy","text_rslp","text_rslps","pre_text_savoy","pre_text_rslp","pre_text_rslps"])
#
#print("Conexão feita com elasticsearch")
#dados = pd.read_csv(r"C:\Users\Flavio\notebooks-pesquisa-1\proposicao-tema-completo.csv", delimiter=',', encoding="utf-8")
#dados.update(dados['txtEmenta'].fillna("Nenhum"))
#dados.update(dados['imgArquivoTeorPDF'].fillna("Nenhum"))
#dados.update(dados['txtExplicacaoEmenta'].fillna("Nenhum"))
#dados.update(dados['txtIndexacao'].fillna("Nenhum"))
#print("Carregou o dataframe")
#dados = [
#    {
#            "content": dados.loc[i,"imgArquivoTeorPDF"],
#            "name":dados.loc[i,"txtNome"],
#
#            "text_savoy": preprocess_savoy(dados.loc[i,"imgArquivoTeorPDF"]),
#            "text_rslp": preprocess_rslp(dados.loc[i,"imgArquivoTeorPDF"]),
#            "text_rslps": preprocess_rslps(dados.loc[i,"imgArquivoTeorPDF"]),
#
#            "pre_text_savoy": preprocess_lowercase_pontuacao_acentuacao_stopword_savoy(dados.loc[i,"imgArquivoTeorPDF"]),
#            "pre_text_rslp": preprocess_lowercase_pontuacao_acentuacao_stopword_rslp(dados.loc[i,"imgArquivoTeorPDF"]),
#            "pre_text_rslps": preprocess_lowercase_pontuacao_acentuacao_stopword_rslps(dados.loc[i,"imgArquivoTeorPDF"])
#
#
#
#
#    }
#   for i in range(len(dados))
#    ]
#print("Já processou, vai indexar..")
#document_store.write_documents(dados,batch_size=9000)
#
#print("Finalizou a 2º")


document_store = ElasticsearchDocumentStore(host='localhost', port=9200, username='', password='', index='experimento_word_n_gram',search_fields=["content","name","text_bigram","text_trigram","text_uni_bi"])
print("Conexão feita com elasticsearch")
dados = pd.read_csv(r"C:\Users\Flavio\notebooks-pesquisa-1\proposicao-tema-completo.csv", delimiter=',', encoding="utf-8")
dados.update(dados['txtEmenta'].fillna("Nenhum"))
dados.update(dados['imgArquivoTeorPDF'].fillna("Nenhum"))
dados.update(dados['txtExplicacaoEmenta'].fillna("Nenhum"))
dados.update(dados['txtIndexacao'].fillna("Nenhum"))

print("Carregou o dataframe")
dados = [
    {
            "content": dados.loc[i,"imgArquivoTeorPDF"],
            "name":dados.loc[i,"txtNome"],
            "text_bigram": preprocess_bigram(dados.loc[i,"imgArquivoTeorPDF"]),
            "text_trigram":preprocess_trigram(dados.loc[i,"imgArquivoTeorPDF"]),
            "text_uni_bi": preprocess_uni_bi(dados.loc[i,"imgArquivoTeorPDF"]),





    }
   for i in range(len(dados))
    ]
print("Já processou, vai indexar..")
document_store.write_documents(dados,batch_size=9000)

print("Finalizou a 3º")

document_store = ElasticsearchDocumentStore(host='localhost', port=9200, username='', password='', index='experimento_pre_word_n_gram',search_fields=["content","name","pre_text_bigram","pre_text_trigram","pre_text_uni_bi"])
print("Conexão feita com elasticsearch")
dados = pd.read_csv(r"C:\Users\Flavio\notebooks-pesquisa-1\proposicao-tema-completo.csv", delimiter=',', encoding="utf-8")
dados.update(dados['txtEmenta'].fillna("Nenhum"))
dados.update(dados['imgArquivoTeorPDF'].fillna("Nenhum"))
dados.update(dados['txtExplicacaoEmenta'].fillna("Nenhum"))
dados.update(dados['txtIndexacao'].fillna("Nenhum"))

print("Carregou o dataframe")

dados = [
    {
            "content": dados.loc[i,"imgArquivoTeorPDF"],
            "name":dados.loc[i,"txtNome"],

            "pre_text_bigram":preprocess_lowercase_pontuacao_acentuacao_stopword_bigram(dados.loc[i,"imgArquivoTeorPDF"]),
            "pre_text_trigram":preprocess_lowercase_pontuacao_acentuacao_stopword_trigram(dados.loc[i,"imgArquivoTeorPDF"]),
            "pre_text_uni_bi":preprocess_lowercase_pontuacao_acentuacao_stopword_uni_bi(dados.loc[i,"imgArquivoTeorPDF"]),

    }
   for i in range(len(dados))
    ]
print("Já processou, vai indexar..")
document_store.write_documents(dados,batch_size=9000)
print("Finalizou 4º")


document_store = ElasticsearchDocumentStore(host='localhost', port=9200, username='', password='', index='experimento_pre_word_n_gram_rslp',search_fields=["content","name","pre_text_bigram_rslp","pre_text_trigram_rslp",
                                                                                                                                                                                      "pre_text_uni_bi_rslp"])
print("Carregou elastic")
dados = pd.read_csv(r"C:\Users\Flavio\notebooks-pesquisa-1\proposicao-tema-completo.csv", delimiter=',', encoding="utf-8")
dados.update(dados['txtEmenta'].fillna("Nenhum"))
dados.update(dados['imgArquivoTeorPDF'].fillna("Nenhum"))
dados.update(dados['txtExplicacaoEmenta'].fillna("Nenhum"))
dados.update(dados['txtIndexacao'].fillna("Nenhum"))
print("Carregou df")
dados = [
    {
            "content": dados.loc[i,"imgArquivoTeorPDF"],
            "name":dados.loc[i,"txtNome"],
            "pre_text_bigram_rslp":preprocess_lowercase_pontuacao_acentuacao_stopword_bigram_rslp(dados.loc[i,"imgArquivoTeorPDF"]),
            "pre_text_trigram_rslp":preprocess_lowercase_pontuacao_acentuacao_stopword_trigram_rslp(dados.loc[i,"imgArquivoTeorPDF"]),
            "pre_text_uni_bi_rslp":preprocess_lowercase_pontuacao_acentuacao_stopword_uni_bi_rslp(dados.loc[i,"imgArquivoTeorPDF"]),
    }
   for i in range(len(dados))
    ]
print("Converteu")
document_store.write_documents(dados,batch_size=9000)

print("Finalizou a 5º")

document_store = ElasticsearchDocumentStore(host='localhost', port=9200, username='', password='', index='experimento_pre_word_n_gram_rslps',search_fields=["content","name","pre_text_bigram_rslps","pre_text_trigram_rslps",
                                                                                                                                                                                      "pre_text_uni_bi_rslps"])
print("Fez a conexao")
dados = pd.read_csv(r"C:\Users\Flavio\notebooks-pesquisa-1\proposicao-tema-completo.csv", delimiter=',', encoding="utf-8")
dados.update(dados['txtEmenta'].fillna("Nenhum"))
dados.update(dados['imgArquivoTeorPDF'].fillna("Nenhum"))
dados.update(dados['txtExplicacaoEmenta'].fillna("Nenhum"))
dados.update(dados['txtIndexacao'].fillna("Nenhum"))
print("Leu DF")
dados = [
    {
            "content": dados.loc[i,"imgArquivoTeorPDF"],
            "name":dados.loc[i,"txtNome"],
            "pre_text_bigram_rslps":preprocess_lowercase_pontuacao_acentuacao_stopword_bigram_rslps(dados.loc[i,"imgArquivoTeorPDF"]),
            "pre_text_trigram_rslps":preprocess_lowercase_pontuacao_acentuacao_stopword_trigram_rslps(dados.loc[i,"imgArquivoTeorPDF"]),
            "pre_text_uni_bi_rslps":preprocess_lowercase_pontuacao_acentuacao_stopword_uni_bi_rslps(dados.loc[i,"imgArquivoTeorPDF"]),
    }
   for i in range(len(dados))
    ]
print("Converteu")
document_store.write_documents(dados,batch_size=9000)
print("Finalizou a 6º")

document_store = ElasticsearchDocumentStore(host='localhost', port=9200, username='', password='', index='experimento_pre_word_n_gram_savoy',search_fields=["content","name","pre_text_bigram_savoy","pre_text_trigram_savoy",
                                                                                                                                                                                      "pre_text_uni_bi_savoy"])
print("Conexão feita com elasticsearch Savoy")

dados = pd.read_csv(r"C:\Users\Flavio\notebooks-pesquisa-1\proposicao-tema-completo.csv", delimiter=',', encoding="utf-8")
dados.update(dados['txtEmenta'].fillna("Nenhum"))
dados.update(dados['imgArquivoTeorPDF'].fillna("Nenhum"))
dados.update(dados['txtExplicacaoEmenta'].fillna("Nenhum"))
dados.update(dados['txtIndexacao'].fillna("Nenhum"))
print("Carragou dataframe")

dados = [
    {
            "content": dados.loc[i,"imgArquivoTeorPDF"],
            "name":dados.loc[i,"txtNome"],

            "pre_text_bigram_savoy":preprocess_lowercase_pontuacao_acentuacao_stopword_bigram_savoy(dados.loc[i,"imgArquivoTeorPDF"]),
            "pre_text_trigram_savoy":preprocess_lowercase_pontuacao_acentuacao_stopword_trigram_savoy(dados.loc[i,"imgArquivoTeorPDF"]),
            "pre_text_uni_bi_savoy":preprocess_lowercase_pontuacao_acentuacao_stopword_uni_bi_savoy(dados.loc[i,"imgArquivoTeorPDF"]),

    }
   for i in range(len(dados))
    ]
print("Já processou, vai indexar..")
document_store.write_documents(dados,batch_size=9000)

print("Finalizou a 7º")
