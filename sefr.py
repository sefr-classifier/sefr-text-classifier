
import numpy as np


class SEFR:
    

    def __init__(self):
        self.weights = []
        self.bias = 0
        


    def fit(self, train_predictors, train_target):
        
        self.dpos = {'key': 1}
        self.dneg = {'key': 1}
        self.dscore = {'key': 1}

        self.numpos = 0
        self.numneg = 0

        """
        This is used for training the classifier on data.
        Parameters
        ----------
        train_predictors : float, either list or numpy array
            are the main data in DataFrame
        train_target : integer, numpy array
            labels, should consist of 0s and 1s
        """
        X = train_predictors
        #X = np.array(train_predictors, dtype="float32")
        

        
        y = train_target
        y = np.array(train_target, dtype="int32")
        

        # pos_labels are those records where the label is positive
        # neg_labels are those records where the label is negative
        
        for i, text in enumerate(X.values):
            l = y[i]
            for token in text.split():
                if (l==1):
                    self.numpos += 1
                    if token in self.dpos:
                        self.dpos[token] = self.dpos[token] + 1
                    else:
                        self.dpos[token] = 1
                    if not token in self.dscore:
                        self.dscore[token] = 0
                if (l==0):
                    self.numneg += 1
                    if token in self.dneg:
                        self.dneg[token] = self.dneg[token] + 1
                    else:
                        self.dneg[token] = 1
                    if not token in self.dscore:
                        self.dscore[token] = 0

        for i in self.dpos:
            self.dpos[i] = self.dpos[i]/self.numpos
            
        for i in self.dneg:
            self.dneg[i] = self.dneg[i]/self.numneg
                                
        
        for i in self.dscore:
            posscore = self.dpos[i] if i in self.dpos else 0
            negscore = self.dneg[i] if i in self.dneg else 0
            score = (posscore - negscore) / (posscore + negscore)
            self.dscore[i]=score                        

        scorepos = 0
        scoreneg = 0            
        for i, text in enumerate(X.values):
            l = y[i]
            if (l==1):
                for token in text.split():
                    scorepos += self.dscore[token] 
            if (l==0):
                for token in text.split():
                    scoreneg += self.dscore[token]
        scorepos /= self.numpos
        scoreneg /= self.numneg
        self.bias = (self.numneg * scorepos + self.numpos * scoreneg) / (self.numneg + self.numpos)  # Eq. 9
                    

    def predict(self, test_predictors):
        """
        This is for prediction. When the model is trained, it can be applied on the test data.
        Parameters
        ----------
        test_predictors: either list or ndarray, two dimensional
            the data without labels in
        Returns
        ----------
        predictions in numpy array
        """
        X = test_predictors
        
        pred = []
        
        for text in X.values:
            score = 0
            for token in text.split():
                if token in self.dscore:
                    score += self.dscore[token]
            if (score-self.bias > 0):
                pred.append(1)
            else:
                pred.append(0)
            
        return pred
