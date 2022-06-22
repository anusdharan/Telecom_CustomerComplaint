import re
import nltk
#import pickle
import warnings
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
warnings.filterwarnings("ignore")
import os

# Now we have a pickled trained model, we can use this for our flask app!
with open(r'C:\Users\Admin\Downloads\Telecome_Complaint\IPNB_files\c1_BoW_Sentiment_Model.pkl', 'rb') as to_read:
       cv= pickle.load(to_read)
with open(r'C:\Users\Admin\Downloads\Telecome_Complaint\IPNB_files\finalized.pickle', 'rb') as to_read:
    classifier= pickle.load(to_read)

def cleanup(new_review):
    #global new_review
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    return new_corpus

def process_text(corpus):
    new_corpus = cleanup(corpus)
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred  = classifier.predict(new_X_test)
    #print(new_y_pred)
    if int(new_y_pred) ==0:
        prediction ='Billing & contact us:(XXX)XXX-8981'
    elif int(new_y_pred) ==1:
        prediction ='Email & contact us:(XXX)XXX-3693'
    elif int(new_y_pred) ==2:
        prediction ='Grievance & contact us:(XXX)XXX-8541'
    elif int(new_y_pred) ==3:
        prediction =' Internet & contact us:(XXX)XXX-8450'
    elif int(new_y_pred) ==4:
        prediction ='Network & contact us:(XXX)XXX-2420'
    elif int(new_y_pred) ==5:
        prediction='Other & contact us:(XXX)XXX-7334'
    else:
        prediction ='Outages & contact us:(XXX)XXX-0211'
    return prediction