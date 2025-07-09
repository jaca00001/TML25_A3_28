## Files and Their Descriptions  

main.py - main code for the defense, loads and trains the Resnet model.

data_sets.py - In here, the given Dataset class is stored.

utils_.py -   This file includes different functions required train and evaluate the model:  
              train: trains the model on either normal or adversarial data.  
              evaluate: returns the accuracy for a given epsilon  
             

attack.py -   In here the different stragtegies to generate adversarial examples are stroed.
              base_attack: capable of performing different attacks based on paramters set, every methhod can either use labels or least likely class.
              fgsm_attack: wrapper of base_attack, implements fgsm 
              r_fgsm_attack: wrapper of base_attack, implements fgsm with random start
              pgd_attack: wrapper of base_attack, implements pgd 
              r_pgd_attack: wrapper of base_attack, implements pgd with random start

## Dependencies

The following Python packages and modules are required to run the scripts:


- numpy
- os
- requests
- matplotlib
- tqdm
- torch
- torchvision
- typing

### Custom modules
- src.attacks
- src.utils
- src.dataset 