import json
import numpy as np

class EarlyStopping:
    def __init__(self, patience, name, is_better_fn):
        self.patience = patience
        self.name = 'main_cost/avg'
        self.is_better_fn = is_better_fn
        self.metric_class_name = is_better_fn.__self__.__class__.__name__
        self.best = None  # best VALIDATION
        self.best_call_counter = 0  # best VALIDATION epoch
        self.best_chpt = None  # address to best VALIDATION checkpoint, if provided
        self.corresponding_test = None  # TEST value for the best VALIDATION
        self.should_stop = False
        self.patience_counter = 0
        self.call_counter = 0
        self.anynan = False
        self.min_delta = 0.05

    def reset_patience(self):
        self.patience_counter = 0

    def reduce_patience(self):
        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            self.should_stop = True

    def __call__(self, vlog, tlog, chpt_str=''):
        if self.should_stop:
            return
        if np.isnan(vlog[self.name]):
            self.anynan = True
            self.reduce_patience()
            return
        if self.best is None:  # keep separate from next condition
            self.best = vlog[self.name]
            self.best_call_counter = self.call_counter
            self.best_chpt = chpt_str
            self.corresponding_test = tlog
            self.corresponding_valid = vlog
            self.reset_patience()
        elif self.is_better_fn(vlog[self.name] + self.min_delta, self.best):
            self.best = vlog[self.name]
            self.best_call_counter = self.call_counter
            self.best_chpt = chpt_str
            self.corresponding_test = tlog
            self.corresponding_valid = vlog
            self.reset_patience()
        else:
            self.reduce_patience()
        self.call_counter += 1
        print('Patience count: ', self.patience_counter)

    def save(self, _file):
        with open(_file, 'w') as f:
            f.write(json.dumps(self.get_status(), indent=4))

    def get_status(self):
        return dict(
            name=self.name,
            best=self.best,
            best_call_counter=self.best_call_counter,
            best_chpt=self.best_chpt,
            corresponding_test=self.corresponding_test,
            corresponding_valid=self.corresponding_valid,
            should_stop=self.should_stop,
            patience_counter=self.patience_counter,
            call_counter=self.call_counter,
            anynan=self.anynan,
            metric_class_name=self.metric_class_name,
        )



class EarlyStoppingVAE:
    def __init__(self, patience=3, min_delta1 = 1, min_delta2 = 0.1):
        self.patience = patience
        self.min_delta1 = min_delta1
        self.min_delta2 = min_delta2

        self.patience_cnt1 = 0
        self.prev_loss_val1 = 200000

        self.patience_cnt2 = 0
        self.prev_loss_val2 = 200000

    def stop(self, loss_val1, loss_val2):
        if(self.prev_loss_val1 - loss_val1>self.min_delta1):
            self.patience_cnt1 = 0
            self.prev_loss_val1 = loss_val1
        else:
            self.patience_cnt1 += 1
            
        if(self.prev_loss_val2 - loss_val2>self.min_delta2):
            self.patience_cnt2 = 0
            self.prev_loss_val2 = loss_val2
        else:
            self.patience_cnt2 += 1

        print('Patience count1, count2: ', self.patience_cnt1, self.patience_cnt2)
        if self.patience_cnt1 > self.patience and self.patience_cnt2 > self.patience :
            return True
        else:
            return False
