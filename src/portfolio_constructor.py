import numpy as np
from scipy.stats import spearmanr

class PortfolioConstructor:
    def __init__(
        self,
        top_k = 50,
        softmax_temp = 0.25,
        transaction_cost_bps = 10,
        debug = False
    ):
        self.top_k = top_k
        self.softmax_temp = softmax_temp
        self.debug = debug
        self.transaction_cost = transaction_cost_bps/10000
        self.prev_pos = None
    
    def softmax(preds,temp):
        temp = max(temp,1e-12)
        x = np.asarray(preds)
        x = x-np.max(x)
        x = x/temp
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-12)

    def construct_portfolio(
            self,
            preds,
            actuals
    ):
        preds = np.array(preds)
        actuals = np.array(actuals)

        order = np.argsort(preds)
        long_idx = order[-self.top_k:]
        bottom_k = len(actuals)-self.top_k
        short_idx = order[:bottom_k]

        long_preds = preds[long_idx]
        short_preds = preds[short_idx]

        long_weights = self.softmax(long_preds,self.softmax_temp)
        short_weights = self.softmax(-short_preds,self.softmax_temp)

        positions = np.zeros(len(actuals),dtype = np.float64)
        if self.prev_positions is None:
            turnover = float(np.abs(positions))
        else:
            turnover = float(np.abs(positions-self.prev_positions).sum())
        self.prev_positions = positions.copy()

        returns = (actuals[long_idx]*long_weights + actuals[short_idx]*short_weights).sum()
        turnover_costs = turnover*self.transaction_cost
        net_returns = returns - turnover_costs

        pred_std = preds.std() 
        ic = float(spearmanr(preds,actuals).statistic) if pred_std > 1e-12 else np.nan
        
        long_hitrate = (np.sign(actuals[long_idx]) == 1).mean()
        short_hitrate = (np.sign(actuals[short_idx]) == -1).mean()
        hitrate = (long_hitrate * self.top_k + short_hitrate*(len(actuals)-self.top_k))/len(actuals)

        if self.debug:
            #TODO add this
            pass

        results = {
            'returns': float(returns),
            'net_returns': float(net_returns),
            'turnover': float(turnover),
            'turnover_costs': float(turnover_costs),
            'spearman_ic': float(ic),
            'hit_rate': float(hitrate)
        }
        return results


        




        

        

    
