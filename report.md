# TML Assignment 3

## Robustness

For this assignment, we were tasked with protecting a Resnet model against adversarial attacks 
while still keeping a relatv high clean accuracy. 

## Data Accessible to the Defender  
- Labeled train Dataset 
- model architecure which can be used

## Approaches Used  
We used standard adversarial training to defend our model against the adversary.
Our design choises will be explained here:

### Why Use Model X?  
We picked Resnet-18 out of the three available models, as we saw in different paper that the depth of the model only plays a minor role 
and Resnet-18 was the one we could experiment the most with due to the smaller size and hence faster training.

### Why Only Train The Model for X Epochs?  
Our best performing model across all three tests was trained for 7 epochs. Even though the loss decreased further the test acccuracy did not change much so
we stopped the training early.

### Why Choose X To Generate Adversarial Samples ?
At the start we used "weaker" methods like FGSM but we quickyl noticed the low performance in the PGD setting so we switched to
experimenting with PGD.

### Which Loss Was Used ?
We tried using only the aversarial samples but this reduced the clean loss too much. 
As a result we use both the clean and aversarial loss to ballance out the accuracies.

## Which Epsilon, ... Was Used ? 
We tried out different values for epsilon until we realized that since we scaled the image by /255 we also need to scacle the epsilon as well.
Next we tried, similiarly to how the timestep in DDPM's is samples, to randomly select our epsilon based on a Normal distribution centered around 8/255.
However experiments have shown that a fixed epsilon performs better, which does not make sense for us in that context.


## Results  


## Observations  


## Other Ideas and Implementation Details  
We tried generating the adversarial samples using a different clean model but it did not work as well.
Tried out one esilon ber epoch, batch and image.