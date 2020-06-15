"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>


---------------------------------------------------
"""

from sklearn.base import BaseEstimator, TransformerMixin

from scipy_modeler._flags.historic_flags import sp_india_hist_flags_keep, \
    sp_india_hist_flags_drop

class SPIndianHistoricFlagsGenerator(BaseEstimator, TransformerMixin):
    """
    Custom transformer for generating historical flags.
    SKlearn (wisely) relies on duck typing over inheritence. So
    all we have to do is define fit and transform and we 
    have our our transformer which will work in a pipeline. 
    
    This uses known columns which are always used to make flag
    variables in the same way (so hard coding is fine and the only way).
    
    Parameters
    ----------

    
    Attributes
    ----------

        
    Notes
    ------


    """
    def __init__(self):
        self.to_keep = sp_india_hist_flags_keep 
        self.to_drop = sp_india_hist_flags_drop
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ###
        for k,v in self.to_keep.items():
            X.set_to_flag(k, value=v, check='equality', 
            drop_original=True)
        for k,v in self.to_drop.items():
            X.set_to_flag(k, value=v, check='equality', 
            drop_original=False)