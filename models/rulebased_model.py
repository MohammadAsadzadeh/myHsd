import pandas as pd
import re
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class RuleBasedModel():
    def __init__(self, offensive_dictionary : pd.DataFrame, ambiguous_dictionary : pd.DataFrame, ner_tokenizer, ner_model, ner_labels, preprocessor):
        self.offensive_dictionary = offensive_dictionary
        self.ambiguous_dictionary = ambiguous_dictionary
        self.ner_tokenizer = ner_tokenizer
        self.ner_model = ner_model
        self.ner_labels = ner_labels
        self.preprocessor = preprocessor

    def offensive_rulebased_model(self, tweet : str):
        for index, row in self.offensive_dictionary.iterrows():
            if row['SearchType'] == 'normal':
                if re.search(row['Word'], tweet) is not None:
                    return 1, row['Word']
            if row['SearchType'] == 'before':
                if re.search("[^\u0622-\u06CC]" + row['Word'], tweet) is not None or re.search("^" + row['Word'], tweet) is not None:
                    return 1, row['Word']
            if row['SearchType'] == 'after':
                if re.search(row['Word'] + "[^\u0622-\u06CC]", tweet) is not None or re.search(row['Word'] + "$", tweet) is not None:
                    return 1, row['Word']
            if row['SearchType'] == 'both':
                if re.search("[^\u0622-\u06CC]" + row['Word'] + "[^\u0622-\u06CC]", tweet) is not None or \
                    re.search("^" + row['Word'] + "[^\u0622-\u06CC]", tweet) is not None or \
                    re.search("[^\u0622-\u06CC]" + row['Word'] + "$", tweet) is not None or \
                    re.search("^" + row['Word'] + "$", tweet) is not None:
                    return 1, row['Word']
        return 0, ""
    
    def ner_rulebased_model(self, tweet : str):
        tokens = self.ner_tokenizer.tokenize(self.ner_tokenizer.decode(self.ner_tokenizer.encode(tweet)))
        inputs = self.ner_tokenizer.encode(tweet, return_tensors="pt")
        outputs = self.ner_model(inputs)[0]
        predictions = torch.argmax(outputs, axis=2)
        predictions = [(token, self.ner_labels[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())]
        for i, (token, token_type) in enumerate(predictions):
            if token_type != 'O' and predictions[i+1][0] in list(self.ambiguous_dictionary.loc[self.ambiguous_dictionary["jointNER"]==1]['Word']):
                return 1, token +" "+ predictions[i+1][0]
        return 0, ""
         
    def __call__(self, tweet : str):
      predicted, keyword = self.offensive_rulebased_model(tweet)
      if predicted == 1:
          return predicted, keyword, "Offensive"
      else:
          predicted, keyword = self.ner_rulebased_model(tweet)
      if predicted == 1:
          return predicted, keyword, "NER"
      else:
        return 0, "", ""
        
    
    def run(self, dataset, mod):
        TP_List = pd.DataFrame(columns=["id", "keyword", "tweet", "model_decision"])
        FP_List = pd.DataFrame(columns=["id", "keyword", "tweet", "model_decision"])
        FN_List = pd.DataFrame(columns=["id", "tweet", "model_decision"])
        TN_List = pd.DataFrame(columns=["id", "tweet", "model_decision"])
        y_pred = []
        y_true = []
        for index, tweetrow in tqdm(dataset.iterrows()):
            inputtweet = tweetrow['tweet']
            
            inputId = tweetrow['id']
            inputtweet = self.preprocessor(inputtweet)
            predicted, foundword, model_decision = self(inputtweet)
            y_pred.append(predicted)
            if mod == "inference":
                continue
            
            inputlabel = tweetrow['label']
            y_true.append(inputlabel)
    
            if predicted == 1 and inputlabel == 1:
                new_record = pd.DataFrame([[inputId, foundword, inputtweet, model_decision]], columns=["id", "keyword", "tweet", "model_decision"])
                TP_List = pd.concat([TP_List, new_record])
            if predicted == 1 and inputlabel == 0:
                new_record = pd.DataFrame([[inputId, foundword, inputtweet, model_decision]], columns=["id", "keyword", "tweet", "model_decision"])
                FP_List = pd.concat([FP_List, new_record])
            if predicted == 0 and inputlabel == 1:
                new_record = pd.DataFrame([[inputId, inputtweet, model_decision]], columns=["id", "tweet", "model_decision"])
                FN_List = pd.concat([FN_List, new_record])
            if predicted == 0 and inputlabel == 0:
                new_record = pd.DataFrame([[inputId, inputtweet, model_decision]], columns=["id", "tweet", "model_decision"])
                TN_List = pd.concat([TN_List, new_record])
        conf = None
        if mod == "eval":
            conf = confusion_matrix(y_true, y_pred)
        return y_true, y_pred, TP_List, FP_List, FN_List, TN_List, conf
    
