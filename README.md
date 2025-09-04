This repository provides code for *The Traveling Bandit: A Framework for Bayesian Optimization with Movement Costs*.


Below is an instruction on how to set up the required environment. First, please open Anaconda Powershell. Locate this folder set it as the workspace by 

```
cd YOUR_PATH\THIS_REPOSITORY
```

Next, create a fresh environment using the following code and respond y to all prompts:

```
conda create --name travelbo python=3.12.11
```

Before activating the new environment, make sure to deactivate the base:

```
conda deactivate
```

Activate the new environment:

```
conda activate travelbo
```

You should now be in the (travelbo) environment. Now, install the required packages:

```
pip install -r requirements.txt
```

To reproduce a figure of a specific test function, first go to the corresponding directory. For example, for Griewank, you will use 

```
cd 2d_Griewank
```


Then, run:

```
python figure.py
```

The output will be saved under 2d_Griewank/Figures folder.
