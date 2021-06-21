from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from pathlib import Path
import dill
nltk.download('stopwords')
nltk.download('punkt')


class NlpModelProvider(object):
    """Generate trained model from the eprovided data frame"""

    stop_words = set(stopwords.words('english'))

    def __init__(self, modelLocation):
        self._modelLocation = modelLocation

    
    def generateModel(self, data):
        #https://www.nltk.org/api/nltk.tokenize.html
        stop_words = set(stopwords.words('english'))

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
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=20)
    
        # Save model. 
        with open(self._modelLocation,'wb') as f:
            dill.dump(model, f)

        #model.save(self._modelLocation)
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
            if w not in self.stop_words:
                filtered_sentence.append(w)
        return filtered_sentence