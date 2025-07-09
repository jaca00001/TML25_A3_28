## Files and Their Descriptions  

MIA_v1.py - main code for the first implementation which loads the dataset, trains the mlp and predicts the membership scores. Run this file for version 1.  

MIA_v2.py - main code for the second implementation of the MIA. This file can be run using multiple console flags to set hyperparamters and returns a csv file containing the membership scores.  

MIA_v3.py - main code for the third implementation of the MIA, run this to perform the attack and save the scores in the csv  

data_sets.py - In here, the given Dataset classes are stored   
               return_private_data_loader: changes the membership to -1 as None caused issues and returns a loader   
               return_public_data_loader: splits the data into subsets for the shadow models, each datapoint should appear in 50% of the datasets.  

utils_v1.py - This file includes different functions required to compute the membership scores for approach one:  
              load_resnet18: loads the pre-trained model with the weights  
              train: trains the shadow models to directly predict membership score.  
              evaluate: returns the tpr for a test set and shadow models  
              predict: given shadow models and unlabeled data returns membership scores  
              compute_losses : given input, computes the loss on the target model  
              compute_tpr_at_fpr: helper function to compute the tpr given true labels and guesses  
              ShadowModel: shadow model class, uses the target model to compute features and uses them to train mlp at predicting membership scores  
              create_shadow_models: creates and returns a number of shadow models  

utils_v2.py - includes code for training the shadowmodel Resnets, the computation of the features,   
the creation of the distributions and the final membership score computation using llr   
              load_resnet18: loads the pre-trained model with the weights  
              return_loaders: prepares and returns the different loaders needed for the attack  
              grad_norm_per_sample: computes the gradnorm, can be used as a feature  
              train_shadow_models: trains shadow models with a Resnet base on its respective loader on the original task using CE loss
              collect_feature_scores: extract features based on the mode from member and non members using the shadow models (loss,entropy,...) 
              fit_kde_distributions: given these features we approximate the distribution using gaussian kde 
              fit_kde: combines the previous two functions and returns the kde's given data and hyperparameters  
              compute_llr_scores: core of this approach, we compute for each private data point the same feature but on the target model and compare the 
                                    result to the tow previously computed kde and calculate the LLr score. The membership is now the sigmoid of it.
              evaluate: returns the tpr for a test set and shadow models  
              predict_llr_private: given shadow models and unlabeled data returns membership scores  

utils_v3.py - similar to utils_v2.py, functions to train the shadow models and compute the membership scores for version 3  
              get_subset_dataloader: given a dataloader, returns a subset dataloader  
              load_resnet18: loads the pre-trained model with the weights  
              return_loaders: prepares and returns the different loaders needed for the attack, slightly changed from v2  
              train_shadow_models: trains shadow models with a Resnet base on its respective loader on the original task, same as v2  
              compute_member_scores: computes the membership score using the shadow models, the private data and a subset of the public data as described in the paper "Low-Cost High-Power Membership Inference Attacks"  
              evaluate: returns the tpr for a test set and shadow models  

## Dependencies

The following Python packages and modules are required to run the scripts:

- argparse
- numpy
- pandas
- requests
- matplotlib
- seaborn
- tqdm
- torch
- torchvision
- scikit-learn
- scipy

### Custom modules
- src.utils_v1
- src.utils_v3
- src.data_sets (including `MembershipDataset`, `return_private_data_loader`, and `return_public_data_loader`)