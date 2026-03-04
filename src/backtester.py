import numpy as np
from scipy.stats import spearmanr
import torch

from data_preparation import FoldData


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
        self.prev_positions = None
    

    def softmax(self, preds, temp):
        temp = max(temp,1e-12)
        x = np.asarray(preds)
        x = x - np.max(x)  # for numerical stability
        exp_x = np.exp(x / temp)
        return exp_x / (exp_x.sum() + 1e-12)  #
    
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

        # Signed positions: +long_weights for longs, -short_weights for shorts
        positions = np.zeros(len(actuals), dtype=np.float64)
        positions[long_idx] = long_weights * 0.5    # 50% gross long
        positions[short_idx] -= short_weights * 0.5 # 50% gross short

        if self.prev_positions is None:
            turnover = 0.0
        else:
            turnover = float(np.abs(positions-self.prev_positions).sum())
        self.prev_positions = positions.copy()

        returns = (actuals[long_idx]*long_weights + (-actuals[short_idx])*short_weights).sum()
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
        #TODO think and add other metrics
        return results

def get_sharpe_ratio(returns, periods_per_year=252):
    """Annualised Sharpe ratio."""
    arr = np.array(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    std = arr.std()
    if std < 1e-12:
        return 0.0
    return float((arr.mean() / std) * np.sqrt(periods_per_year))


def evaluate_model_per_fold(
        model,
        top_k,
        softmax_temp,
        fold_data,
        transaction_cost_bps = 25,
        debug = False,

):
    model.eval()
    fold_returns = []
    fold_spearman_ic = []
    fold_turnover = []
    fold_hit_rate = []
    fold_metrics = []
    Portfolio = PortfolioConstructor(top_k,softmax_temp,transaction_cost_bps,debug)
    curr_portfolio_value = 1.0
    min_portfolio_value = 0.0
    max_drawdown = 0.0
    max_portfolio_value = 1.0
    with torch.no_grad():
        for i in range(len(fold_data.test_idx)):
            x = fold_data.X_test_tensor[i]       # [N, F]
            N, F = x.shape                        # fixed: fold_data has no .shape
            
            preds = model(x).detach().cpu().numpy()  # raw logits → numpy
            actuals = fold_data.y_test[i]         # [N] numpy
            metrics = Portfolio.construct_portfolio(preds, actuals)
            curr_portfolio_value *= (1.0 + metrics['net_returns'])  # fixed: compound correctly
            min_portfolio_value = min(min_portfolio_value, curr_portfolio_value)
            max_portfolio_value = max(max_portfolio_value, curr_portfolio_value)
            max_drawdown = max(max_drawdown,1.0 - curr_portfolio_value/(max_portfolio_value + 1e-12))  
            fold_returns.append(metrics['net_returns'])
            fold_spearman_ic.append(metrics['spearman_ic'])
            fold_turnover.append(metrics['turnover'])
            fold_hit_rate.append(metrics['hit_rate'])
            fold_metrics.append(metrics)          

    avg_return = np.mean(fold_returns)
    avg_spearman_ic = np.nanmean(fold_spearman_ic)
    avg_turnover = np.mean(fold_turnover)
    avg_turnover_costs = avg_turnover*Portfolio.transaction_cost
    avg_hit_rate = np.mean(fold_hit_rate)
    sharpe_ratio = get_sharpe_ratio(fold_returns, 252)
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Avg Return: {avg_return:.6f} | Hit Rate: {avg_hit_rate:.4f} | Spearman IC: {avg_spearman_ic:.4f}")
    print(f"Avg Turnover: {avg_turnover:.4f} | Avg Turnover Cost: {avg_turnover_costs:.6f}")
    print(f"Final Portfolio Value: {curr_portfolio_value:.4f} | Max Drawdown: {max_drawdown*100:.2f}%")
    results = {
        'avg_return': float(avg_return),
        'avg_spearman_ic': float(avg_spearman_ic),
        'avg_turnover': float(avg_turnover),
        'avg_turnover_costs': float(avg_turnover_costs),
        'avg_hit_rate': float(avg_hit_rate),
        'sharpe_ratio': float(sharpe_ratio),
        'final_portfolio_value': float(curr_portfolio_value),
        'max_drawdown_pct': float(max_drawdown*100),
        'fold_returns': fold_returns
    }
    return results








            







        

        

    
