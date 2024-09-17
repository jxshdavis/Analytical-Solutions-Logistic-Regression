# Mentor Meeting Notes 09.11.2024
Student: Josh Davis
Mentor: Meng Li

## Meeting Outline

Our meeting consited of a discussion of our existing project, restating of our main goals, and laying out actionable steps.

## Project Refresher

Our research currently has
- a closed form estimator for the logistic regression problem on binary data
- a mathematical analysis of asymptotic properties
- a simulation and comparison with the Maximum Likelihood Estimator
- a comparison on a real breast cancer dataset

## Updated Project Goals

We discussed scenarios where our current model struggles. Currently, these are when data points have a homogeneous response (all zeros or all ones) and low or zero row counts. We hope make progress on these issues durring the capstone.

Our main goal is to encoperat LeCams one-step update procedure to our current analytic estimator. With this, we hope to upgrade to a model which can always use all avaiable data (unlike our current model). This will involve:

1. Gaining a solid understanding of the one-step update and how to apply it to our model 
2. Implemeting the one-step update in Python and running simulations.
3. Comparing updated model accurary and confidence intervals to current estimator and MLE.
4. Applying the theoretical results of LeCams method to our scenario.



## Main ToDos

1. Read one-step background questions.
2. Write down the equation for the update applied to our estimator.
3. Implement the one step update.
4. Make sure the user has the option to Remove rows below count k and delete homogenous rows.
5. Update Simulations.
6. Decide out what journal I want to submit to.









