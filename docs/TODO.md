The good:

- dummy version of adaptive weighted ensemble (ma cross + rsi grid search and ga weights with adaptive regime switching (loads best params and weights per regime)) is basically working
- event driven and config stuff is looking thick, solid, tight

The bad:

- code is sloppa, especially the optimizer runners + ensemble_strategy.py




- need to clean codebase and documentation before this breaks
  - refactor optimizers
  
- GA shit is STILL not working
- need analytics display
- need options and API stuff working 
- different wieght logic in the joint optimizer, is this a problem?
- why does the rulewise optimization process have weights at all? it should be running a backtest for only one rule, but currently has weights for both ma and rsi
- signal strength in ensemble strategy should be a job for the risk manager or some other signal processor, should be refactored out 
- ensemble strategy is a shitshow and orchestrates too many processes 
- documentation is unweildy and possibly out of date, needs to be updated and consolidated 
- no bootstrap or container/DI stuff yet 
- lacking granular contorl over log output, reliant on grep voodoo to pull what i need out of massive logs 
- not sure if weights are being applied dynamically like parameters
- why does joint optimization not outperform rulewise? I know there's overfitting possiblity, but still
