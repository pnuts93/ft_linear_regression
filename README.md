# ft_linear_regression
A python project that implements linear regression with gradient descent from scratch

## Introduction
The project is based on a [42](https://42berlin.de/) subject from the advanced curriculum. The goal is to make a linear regression model based on the gradient descent method that tries to predict the price of a car based on its mileage. It includes two parts: `train.py` and `predict.py`.  
1. `train.py` reads into `data/data.csv` and trains the linear regression model. When the training has been completed, the program writes the calculated coefficients in `model.json`.
2. `predict.py` accepts an integer as argument ( the mileage of a car ), reads `model.json` and uses the coefficients to predict the price for that car. 

## Features
The `train.py` program creates an animation with Matplotlib to show how the model "learns" over the iterations. The output is shown below:

![animation](gd.gif)

## Issues
The subject requires to use the following formulas:
$$ \theta _0 = learningRate * \frac{1}{m} \sum_{i=0}^{m - 1} (estimatedPrice(mileage[i]) - price[i])  $$
$$ \theta _1 = learningRate * \frac{1}{m} \sum_{i=0}^{m - 1} (estimatedPrice(mileage[i]) - price[i]) * mileage[i] $$

Although this does not change substantially the formulas that can be obtained via partial derivatives of the mean square error formula ( the one that should be used for gradient descent ), I personally found it rather confusing considering that no reference was given to what motivated the use of this simplified formula. For any other student encountering this issue (or for anyone wondering why I have used this specific formula) here is the information I could gather:
* In this simplified formula the sign of the fraction and its nominator are changed, which should not radically change the final result because
    * Changing the sign changes the "direction" in which the lowest point of the error curve is searched
    * Changing the nominator simply impacts the learning rate, doubling it
* In the equation of a line $ y = mx + b $
    * $ \theta _0 $ corresponds to $ b $
    * $ \theta _1 $ corresponds to $ m $


## Method
The model is trained on a dataset scaled with the min-max scaling method, and the coefficients in the output are re-scaled to the original dimensions by using the opposite process.  
The model defaults to a learning rate of 0.05 and iterates over 1000 epochs. The optimal number of epochs actually falls a bit over one half of the default, but mostly due to aestetich reasons I extended it to 1000 (I wanted a longer MSE curve to show that the curve actually does not keep decreasing after reaching 600 epochs)
