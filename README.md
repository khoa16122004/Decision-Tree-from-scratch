## Decision-Tree-from-scratch

## Using ID3 algorithm to built Decision Tree from scratch

This repo using ID3 to find the best-split of Decision Tree

![Decision Tree](https://cdn.educba.com/academy/wp-content/uploads/2019/11/Create-Decision-Tree-1.png)

## The definition of Suprise of the dataset

This is the metric that evaluate the **Inpeture** of the dataset by their label.

The higher Inpeture is, the higher Suprise is, that mean the dataset is difficul to predict

We can basicly guess that if the probalities of a event is low, the more suprise we get and contrast

So the probalities is inverst with Inpeture, and we have some metrics show that theory:

* Entropy
* Gini
* Log_loss (or Cross_entropy)

### Gini index
Gini(S) = 1 − ∑_ip_i^2

where:
* S is the dataset
* pi is the proportion of label i in dataset S

### Log loss
Log_loss(S, y) = −∑_i y_i log_2(p_i)

where:
* S is the dataset
* y is the actual label of the data
* pi is the proportion of label i predicted for the data
