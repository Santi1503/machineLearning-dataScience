from html.parser import HTMLParser
import email
import string
import nltk
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
nltk.download('stopwords')

# Esta clase facilita el procesamiento de los correos electrónicos con código HTML
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)
    
def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

class Parser:
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)

    def parse(self, email_path):
        """Parse an email"""
        with open(email_path, errors='ignore') as e:
            msg = email.message_from_file(e)
        return None if not msg else self.get_email_content(msg)
    
    def get_email_content(self, msg):
        """Get the content of an email"""
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(), 
                                   msg.get_content_type())
        content_type = msg.get_content_type()
        return {"subject": subject, 
                "body": body, 
                "content_type": content_type}
    
    def get_email_body(self, payload, content_type):
        """Get the body of an email"""
        body = []
        if type(payload) is str and content_type == "text/plain":
            return self.tokenize(payload)
        elif type(payload) is str and content_type == "text/html":
            return self.tokenize(strip_tags(payload))
        elif type(payload) is list:
            for p in payload:
                body += self.get_email_body(p.get_payload(), 
                                            p.get_content_type())
        return body
    
    def tokenize(self, text):
        """Tokenize a string"""
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]
    
"""inmail = open("../machineLearning-dataScience/datasets/trec07p/data/inmail.1").read()
print("----------------------------------- MAIL ------------------------------------")
print(inmail)"""

p = Parser()
p.parse("../machineLearning-dataScience/datasets/trec07p/data/inmail.1")

"""index = open("../machineLearning-dataScience/datasets/trec07p/full/index").readlines()
print("-------------------------------------- INDEX -----------------------------------")
print(index)"""

def parse_index(path_to_index, n_elements):
    ret_indexes = []
    index = open(path_to_index).readlines()
    for i in range(n_elements):
        mail = index[i].split("../")
        label = mail[0]
        path = mail[1][:-1]
        ret_indexes.append({"label":label, "email_path":path})
    return ret_indexes

def parse_email(index):
    p = Parser()
    pmail = p.parse(index["email_path"])
    return pmail, index["label"]

"""indexes = parse_index("../machineLearning-dataScience/datasets/trec07p/full/index", 10)
print("----------------------------------- INDEXES ------------------------------------")
print(indexes)"""

index = parse_index("../machineLearning-dataScience/datasets/trec07p/full/index", 1)
open(index[0]["email_path"]).read()

mail, label = parse_email(index[0])
print("El correo es: ", label)
print(mail)

""" Usando VECTORIZER
prep_email = [" ".join(mail['subject']) + " ".join(mail['body'])]

vectorizer = CountVectorizer()
X = vectorizer.fit(prep_email)

print("Email: ", prep_email, "\n")
print("Caracteristicas de entrada: ", vectorizer.get_feature_names_out())

X = vectorizer.transform(prep_email)
print("\nValues: \n", X.toarray())
"""

def create_prep_dataset(index_path, n_elements):
    X = []
    Y = []
    indexes = parse_index(index_path, n_elements)
    for i in range(n_elements):
        print("\rParsing email: ", i, " "*5, end='')
        mail, label = parse_email(indexes[i])
        X.append(" ".join(mail['subject']) + " ".join(mail['body']))
        Y.append(label)
    return X, Y

"""
#Entrenamiento con 100 correos
X_train, Y_train = create_prep_dataset('../machineLearning-dataScience/datasets/trec07p/full/index', 100)
print("----------------------------------- TRAIN ------------------------------------")
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)

print(X_train.toarray())
print("\nFeatures: ", len(vectorizer.get_feature_names_out()))

df = pd.DataFrame(X_train.toarray(), columns=[vectorizer.get_feature_names_out()])
print(df)

clf = LogisticRegression()
clf.fit(X_train, Y_train)

X, Y = create_prep_dataset('../machineLearning-dataScience/datasets/trec07p/full/index', 150)
X_test = X[100:]
Y_test = Y[100:]

X_test = vectorizer.transform(X_test)

Y_pred = clf.predict(X_test)
print("\n---------------------- PREDICCION ---------------------")
print("Prediccion: \n",Y_pred)
print("\nEtiquetas reales: \n", Y_test)

print('Accuracy: {:.3f}'.format(accuracy_score(Y_test, Y_pred)))
"""

#Entrenamiento con 12000 correos
print("+---------------------------------+")
print("| Entrenamiento con 12000 correos |")
print("+---------------------------------+")
X, Y = create_prep_dataset('../machineLearning-dataScience/datasets/trec07p/full/index', 12000)
x_train, Y_train = X[:10000], Y[:10000]
X_test, Y_test = X[10000:], Y[10000:]

print("\n+-------------------------+")
print("| Entrenamiento terminado |")
print("+-------------------------+")
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(x_train)
clf = LogisticRegression()
clf.fit(X_train, Y_train)

print("+-----------------------+")
print("| Calculando prediccion |")
print("+-----------------------+")
X_test = vectorizer.transform(X_test)
Y_pred = clf.predict(X_test)
print("+---------------------------------+")
print('Accuracy: {:.3f}'.format(accuracy_score(Y_test, Y_pred)))
print("+---------------------------------+")
