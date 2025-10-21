This is the code for the paper {Disentangling Confounders via Counterfactual Interventions for Fair Recommendations}.

For MF Model:

```bash
cd FairDC/FairDC_aMF_ML
python FairMF_main.py
```

If CUDA out of memory, try samll value of K:
```
python FairMF_main.py --K 30
```

For GCCF Model:

```bash
cd FairDC/FairDC_bGCCF_ML
python FairGCCF_main.py
```

For NCF Model:

```bash
cd FairDC/FairDC_cNCF_ML
python FairNCF_main.py
```
