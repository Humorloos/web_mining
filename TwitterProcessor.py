import pandas as pd
import re
import sys 

class TwitterProcessor: 
    
    def token_user(self, inpt):
        if isinstance(inpt, str):
            return re.sub(r'@[^\s]*', 'USERNAME', inpt)
        elif isinstance(inpt, pd.Series):
            return inpt.apply(lambda x : re.sub(r'@[^\s]*', 'USERNAME', x))
        else:
            print("Input should be a string or a pandas series")
            sys.exit()
        
        
    def token_url(self, inpt):
        if isinstance(inpt, str):
            return re.sub(r'http:[^\s]*', 'URL', inpt)
        
        elif isinstance(inpt, pd.Series):
            return inpt.apply(lambda x : re.sub(r'http:[^\s]*', 'URL', x))
        
        else:
            print("Input should be a string or a pandas series")
            sys.exit()
        
    def reduce_letters(self, inpt):
        #TODO
        pass
    
    def remove_false_predictors(self, inpt, predictor_list):
        #TODO
        pass
    
    def make_lower(self, inpt):
        if isinstance(inpt, str):
            return inpt.strip().lower()
        elif isinstance(inpt, pd.Series):
            return inpt.apply(lambda element: element.strip().lower())
        else:
            print("Input should be a string or a pandas series")
            sys.exit()
    
