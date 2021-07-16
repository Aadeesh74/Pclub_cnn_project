hackathon2.ipynb is the file where I firstly used my alexnet implemented model with that i got 65.8% accuracy on testing dataset and then I used normalization layers and increased learning rate of Adam to 3 times so my accuracy went upto 77.7%.

In hackthon2(1).ipynb I changed optimizer from Adam to SGD and removed the dropout layers and tested model on 2 different learning rates without L2 regularization as it gave poor results and so i got accuracy of 79.7 and 68.3 while using lr of 0.05 and 0.06 respectively.

In hackathon2(2).ipynb I used xavier weight initialization technique and introduced L2 regularization in Adam to get an accuracy of 79.5 on testing dataset.
