from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from pathlib import Path
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class NlpModelProvider(object):
    """Generate trained model from the eprovided data frame"""

    stop_words = stopwords.words('english')
    tagged_data = None

    def __init__(self):
        self._modelLocation = "similar_sentence.model"
   
    def generateModel(self, data):
        #https://www.nltk.org/api/nltk.tokenize.html
        stop_words = stopwords.words('english')
        stop_words.extend(['-'])

        tagged_data = [TaggedDocument(words=self.__cleanTokens(word_tokenize(_d.lower())), tags=[str(i)]) for i, _d in enumerate(data)]

        # hyper parameters
        #https://radimrehurek.com/gensim/models/doc2vec.html
        max_epochs = 500
        vec_size =200
        alpha = 0.03
        minimum_alpha = 0.0025
        reduce_alpha = 0.0002

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha, 
                        min_alpha=minimum_alpha,
                        dm =1,#distributed memory (PV-DM) 
                        min_count=1,#very critical, if min is 2 ormore the result is inacurate
                        workers=4)
        model.build_vocab(tagged_data)

        # Train the model based on epochs parameter
        for epoch in range(max_epochs):
            model.train(tagged_data, total_examples=model.corpus_count, epochs=20)
    
        # Save model. 
        model.save(self._modelLocation)
        return model

    def getModel(self):
        modelPath = Path(self._modelLocation)

        if modelPath.is_file():
            return Doc2Vec.load(self._modelLocation)
        else:
            return None

    def __cleanTokens(self, word_tokens):
        filtered_sentence = [w for w in word_tokens if not w.lower() in self.stop_words]
        filtered_sentence = []
 
        for w in word_tokens:
      #Clean up all numbers and words with numbers but keep 365 because it has meaning to us, this increases the accuracy rate by 2-3%
            if re.match('^((?!365)\w*[0-9]\w*)$', w):
                continue
            else:
                filtered_sentence.append(w)
        return filtered_sentence