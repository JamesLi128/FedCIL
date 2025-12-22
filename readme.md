# TODO List

## Find ways to improve the current vinella FCIL

- GAN lr didn't work. 
- More complex classification head.
- server replay
- Concatenate label embed to z instead of add to z
- ACGAN integration.
- ImageNet-family dataset integration
- better result analytics. 
    1. average accuracy (pick the highest for each task and then average)
    2. average forgetting (this needs the accuracy of all the global rounds for each task)
    3. for each known class, generate 4 images and put them in one big image with 1 row and 4 columns, each image takes a column place
    4. 


- Understand how FedGTEA worked and plot a roadmap
- Code the initial version of FedGTEA


## Experiment Results
- GAN-lr and lr does not affect performances greatly, currently it's better to set them larger both at 1e-3
- replay consolidation on server works, already seen 10% accuracy gain on the second task with server steps 5 and server lr 1e-3
- multilayer classification head doesn't work better than singlehead, at least for embed dim = 128