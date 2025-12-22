# TODO List

## Find ways to improve the current vinella FCIL

- GAN lr didn't work. 
- More complex classification head.
- server replay
- Concatenate label embed to z instead of add to z
- ACGAN integration.
- ImageNet-family dataset integration


- Understand how FedGTEA worked and plot a roadmap
- Code the initial version of FedGTEA


## Experiment Results
- GAN-lr and lr does not affect performances greatly, currently it's better to set them larger both at 1e-3
- replay consolidation on server works, already seen 10% accuracy gain on the second task with server steps 5 and server lr 1e-3
- multilayer classification head doesn't work better than singlehead, at least for embed dim = 128