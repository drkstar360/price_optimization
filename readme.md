# Test Task (Hyatus)
## Task:
Given the target 30 × 30 stay‑price matrix in the CSV above, compute:<br/>
• A vector of 30 nightly base rates.<br/>
• For each night, eight discount percentages corresponding to the stated cut‑offs.<br/>
goal is to minimise the discrepancy between the stay prices produced by the PMS rule engine and the target matrix.


## Required Explanation:
Explanations are in [optimization_explanation.md](optimization_explanation.md)

## How to run the script:
1. Install python 3.12
2. create a virtual environment and activate it.
3. install dependencies by 
```commandline
pip install -r requirements.txt
```
4. run the script by
```commandline
python rate_optimizer.py
```