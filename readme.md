# TODO List

## Find ways to improve the current vinella FCIL

- Concatenate label embed to z instead of add to z
- Concatenate one-hot label to z instead of embeddings
- fight-forgetting: penalize excessive update by penalizing gradient norm $\gamma \|g\|$ in the loss function (this seems not working very well)
- fight-forgetting: EWC (seems not working for FCIL)
- fight-forgetting: knowledge-distillation (seems working the best)

- better result analytics. 
    1. average accuracy (pick the highest for each task and then average)
    2. last task accuracy (the highest test accuracy achieved in the last task training)
    3. forgetting measure (this needs the accuracy of all the global rounds for each task)
    4. backward transfer and forward transfer
    5. for each known class, generate 4 images and put them in one big image with 1 row and 4 columns, each image takes a column place, put them in a folder under each hydra sweep configuration folder


- Understand how FedGTEA worked and plot a roadmap
- Code the initial version of FedGTEA


## Experiment Results
- GAN-lr and lr does not affect performances greatly, currently it's better to set them larger both at 1e-3
- replay consolidation on server works minorly, already seen 10% accuracy gain on the second task with server steps 5 and server lr 1e-3, but not for last tasks
- multilayer classification head doesn't work better than singlehead, at least for embed dim = 128
- cat embedding didn't work very well
- implemented a general framework to load any dataset by configuring the hydra yaml file
- Curernt framework is already ACGAN, only separated the classification head as IncrementalNet.