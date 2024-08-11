# self-learning-AI

<h3>Teaching myself AI and documenting the process</h3>
<h2>Day 1</h2>
Learnt what a Perceptron is, how it is defined, and the math behind a perceptron. Implemented a basic Perceptron in python and used it to fit data from the iris flowers dataset and see how the perceptron converges to a decision boundary. <br>
<img width="553" alt="Screenshot 2024-07-12 at 11 39 09 AM" src="https://github.com/user-attachments/assets/28b5507c-b3f2-40bf-af29-6784a2148912"> <br>
<h2>Day 2</h2>
Learnt what a ADAptive LInear NEuron (adaline) is, how it is defined, the math behind it, and the math behind **batch gradient descent** using an **Identity** activation function and **MSE loss**. Implemented a basic adaline in python and used it to fit data from the iris flowers dataset and looked at how the learning rate affects algorithm convergence. <br>
<img width="836" alt="Screenshot 2024-07-13 at 11 27 32 PM" src="https://github.com/user-attachments/assets/9a20e8cf-4bd9-4922-93cd-f374a60d9055"> <br>
<h2>Day 3</h2>
Derived the mathematical background for a perceptron and Adaline by hand. Learnt about feature scaling and one feature scaling technique — standardization. Fixed a bug about the predictions made by Adaline that I had overlooked earlier. Used scaled features to train an Adaline and compared convergence of the Adaline before and after. <br>
<img width="676" alt="Screenshot 2024-07-21 at 5 15 54 AM" src="https://github.com/user-attachments/assets/2308488a-5498-4180-ae71-bc3de2846fa1"><br>
<h2>Day 4</h2>
Implemented the stochastic gradient descent algorithm for an Adaline. Saw how it affects the convergence rate (Adaline reached convergence faster now). Also implemented online learning using the SGD Adaline<br>
<img width="524" alt="Screenshot 2024-07-26 at 2 17 30 AM" src="https://github.com/user-attachments/assets/aca573ac-eaf6-4add-88a5-07db4bd162a4">
<br>
<img width="521" alt="Screenshot 2024-07-26 at 2 17 36 AM" src="https://github.com/user-attachments/assets/f8d04dbc-049e-43ab-a7de-417bf4c0ad03"><br>
<h2>Day 5</h2>
Used scikit-learn's implementation of a Perceptron and tried to fit it so it could classify points into three different classes. The classes were not lineary separable, so Perceptron did not reach convergence (one of the limitations of perceptrons). Also learned about train-test-splitting.<br>
<img width="752" alt="Screenshot 2024-07-30 at 4 28 11 AM" src="https://github.com/user-attachments/assets/880c8682-b15e-44b3-b462-881fae52822b">
<br>
Learned how logistic regression is modelled using logits and derived by hand the mathematics behind the working of logistic regression, including the maximun likelihood and log maximum likelihood, loss function, and gradient descent along the loss function (and ascent). Examined the behaviour of the loss function. Implemented a simple logistic regression algorithm that uses full batch gradient descent to learn. <br>
<img width="754" alt="Screenshot 2024-07-30 at 4 28 18 AM" src="https://github.com/user-attachments/assets/7c1912d6-adf9-4d53-b901-ed5899724083">
<img width="763" alt="Screenshot 2024-07-30 at 4 28 25 AM" src="https://github.com/user-attachments/assets/d23ec8ea-090d-48ac-b03c-2d46c89ba45a">
<br>
<h2>Day 6</h2>
Used scikit-learn's implementation of Logistic Regression to train it on the iris-flower dataset and make predictions on three class labels using OneVsRest classification. Derived the loss function when we add a regularization term to it, and explored the effect of increasing the regularization term on weights. <br>
<img width="753" alt="Screenshot 2024-08-11 at 6 31 38 AM" src="https://github.com/user-attachments/assets/1103155e-61c3-4d0e-bbda-21274f5900ac">
<img width="674" alt="Screenshot 2024-08-11 at 6 32 31 AM" src="https://github.com/user-attachments/assets/6a3d7eba-9610-4f66-8965-6441c921fc4c">
