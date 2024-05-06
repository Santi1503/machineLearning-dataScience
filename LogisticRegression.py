from html.parser import HTMLParser
import email
import string
import nltk
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
    
inmail = open("../machineLearning-dataScience/datasets/trec07p/data/inmail.1").read()
print("----------------------------------- MAIL ------------------------------------")
print(inmail)

p = Parser()
p.parse("../machineLearning-dataScience/datasets/trec07p/data/inmail.1")

index = open("../machineLearning-dataScience/datasets/trec07p/full/index").readlines()

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

indexes = parse_index("../machineLearning-dataScience/datasets/trec07p/full/index", 10)
print("----------------------------------- INDEXES ------------------------------------")
print(indexes)