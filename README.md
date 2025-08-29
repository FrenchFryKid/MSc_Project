The repository contains 10 files. 8 of them are used for data collection. The 2 files resultsCNN.ipynb and resultsMLP.ipynb are used to analyze the data gained from the 8 files and plot graphs.

launchcnn.sh, launchcnnEG.sh and launchcnnEGBias.sh are shell script files that run the file launchcnn.py but parses in different parameters.
launchmlp.sh, launchmlpEG.sh and launchmlpEGbias.sh are shell script files that run the file launchmlp.py but parses in different parameters.

For CIFAR10, it is downloaded with: torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False)
For MNIST, it is downloaded with: torchvision.datasets.MNIST(root=root_dir, train=True, download=True)
