# TML Assignment 3

## Robustness

For this assignment, we were tasked with protecting a Resnet model against adversarial attacks 
while still keeping a relatv high clean accuracy. 

## Data Accessible to the Defender  
- Labeled train Dataset 
- model architecure which can be used

## Approaches Used  
We tried out various methods to generate adversarial samples for the adversarial training but ended up using PGD.


### Why Only X Shadow Models?  
For version 2, the number of shadow models surprisingly did not have a big impact. We achieved the best results with 3–5 shadow models after testing.  
Version 3, however, improved with more shadow models, but compute time also increased. Here, we used 10–50 shadow models per run.

### Why Only Train the Shadow Models for X Epochs?  
For version 2, training for too many epochs led to overfitting and a lower TPR at FPR = 0.05.  
Our best result was achieved using only 5–20 epochs.  
Version 3 was trained longer — up to 40 epochs — which still led to good results.

### Why Use X as the Predictor for the LLR?  
We tested multiple combinations of features and found that using loss as a feature worked best.  
We also used entropy, confidence, gradient norm, and a linear combination of all features.  
Optimizing the weights of the linear combinations using HPO resulted in only loss being selected.

## Results  
The above approach results in a TPR@FPR=0.05 of 0.093 and an AUC of 0.64.  
Our solution ranked 30th on the leaderboard.

## Observations  
At the start, we tried to visualize both members and non-members using PCA and t-SNE and noticed that both overlapped.  
This was confirmed when plotting the distribution of features like loss, entropy, and confidence of members and non-members.  
We tried to separate the two by looking at `feature^exp`, which increased the TPR but only for extreme points.  
Most points in the middle were pushed toward 0.5.  
This only worked for loss, as the other distributions were too similar. Without this exponent, the score would not have been possible, and other methods might have been better.

## Other Ideas and Implementation Details  
- Adding exponents to the features used for LLR — lower exponents increased TPR but caused more overlap in the middle of the two distributions  
- Testing different architectures  
- Tuning necessary parameters using HPO  

