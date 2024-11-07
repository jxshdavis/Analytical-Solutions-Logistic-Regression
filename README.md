# Analytic Logistic Regression


## 09.11.2024

TODO:
1. Figure out what journal I want to submit to.
2. Read one-step background questions.
3. Write down the equation for the update applied to our estimator.
4. Implement the one step update.
5. Make sure the user has the option to Remove rows below count k and delete homogenous rows.
6. Update Simulations.
7. Fix the hard coded "alternate" method where we either drop homogeneous rows or estimate them with $1-\frac{1}{2N_m}$ or $\frac{1}{2N_m}$.
8. Change the .transform_response() function to a .fit() named function.

## 09.17.2024 with McGuffey

- Next time chunk is to make as much technical progress as possible
- TA workshops

- Feedback on Proposal
- Emphasis of the upcoming workshop is for writing introduction

Next Check In: How much progress do I want to encoreprare for my pleminary draft. 
- If Capstone Report: less emphasis of work which has been done (motivation for logistic regression plus one step update, and summary of current work)
- If for journal:

Capstone
- some techinal results
- submissions include cover sheet (like a guide to what is new from the capstone -> techincal progress and writing progress)

ToDo:

1. Joint meeting with McGuffey and Li about our end goals (schedule with Dr. Li)


## 09.18.2024

### Updates
1. Read through LeCam Made Simple paper and took light notes on first section. I don't understand most of it right now.
2. Wrote down the one step update on paper. I still don't fully understand the assumptions needed to use it but I'm just treating it as a Newton's method step right now.
3. Got basic one step upworking working in code with synthetic examples.
4. Fixed issues causing MLE to not compute simulation (skip over cases where data is degenerate)


## 09.18.2024
Meeting Notes:

What we envision for the paper at a high level
1. Presentation can be more of an academic presentation
2. Apply for the distinction thing when it comes out.

One-step
1. Adjust the tone - what is the primary approach we want to advocate. In some scenarios we reccomend the one-step and in others we reccomend the gamma.
2. Try unbalanced design matrix for comparing the two approaches.

### TODO:

1. Find a fixed data set to illustrate the behavior observed in first test of all three methods were analytic estimators had better MSE to true beta.
2. Fix the hard coded "alternate" method where we either drop homogeneous rows or estimate them with $1-\frac{1}{2N_m}$ or $\frac{1}{2N_m}$.
   -** We want to compare 4 different methods now. Make that happen with a clean comparison visualization.**
4. Change the .transform_response() function to a .fit() named function.
5. Run experiments on an unbalaced design matrix (missing rows / low row counts for some rows).
6. Update real data experiments.
7. Spread out histograms in simulation plot.
   
   

### Udates:
1.


### Udates:

1. Figured out Journal and began formatting abstract and intro.
2. Expanded the introduction for draft.
3. Found the data set which illustrates improved performance of analytic estimators.
4. Spread out histograms in simulation plot.
5. Fixed the hard coded "alternate" method... now a model parameter
6. Found new case where pseudo inverse has to be used.
7. Moved all text to Biometrika format and did big set of revisions.
8. Added a new section 2.2 containing a new theorem MLE equivalence. This generalizes the example in 2.1 nicely.
9. Have three examples of single data sets and then the application to the breast cancer dataset.



   
#10.05.2024

###TODO:
1. Run special example where pseudo-inverse has to be used.
2. Add new tests to the document to show Dr. Li in next meeting / email update.
3. Run real data experiments with new table thing.
4. Update manuscript with new cases found with pseudo inverse.
5. Generalize Lemma 1 to only cover the case where $M \geq p+1$ (I later realized this was not possible ie there are examples when M=p+1 but the matrix is not injective.


   
## 10.09.2024


Pre Meeting Plan
1. Four analytic estimators and plots.
2. Defining a new analytic estimator when $\tilde X$ does not exist or when $\tilde X^T \tilde X$ is not invertible.
3. New work on when Gamma is the MLE.

Meeting Notes
1. Since last meeting, I formally descirbed four variations of the analytic estimator. I described the 4 variations to Dr. Li and he asked clarifying questions.
2. Discussed looking at other logistic regression papers to see how other people set up experiments.
3. Run Experiemnts with more "tame" $\beta$s say in the interval $[-1,1]$.
4. For experiments that use a test train split, do 50/50 and 80/20.
5. Look into the amazon dataset.

## 10.16.24

TODO:
1. Reduce dimension fo breast cancer data set and re-run experiments.
2. Look up Epithial cell size variable and see if it is mentioned in bio literature.
3. Try to find a transformation similar to the t-test for gamma.

UPDATES:

1. Applied random forest to reduce amazon dataset to 5 predictors.
2. Found one paper on Epithial cells.
3. Did a lot of work for the closed form distribution but it didn't go anywhere useful.
   
## 11.06.24

1. Breast cancer Epithial cell size
  - establish if the paper I found can really be used as a comparison
  - find more paper(s) on this topic
2. Brush up algorithm 6.
3. Get a time comparison where both methods can work directly with tilde X.
4. Change MC Plots back to over lapping bars (different opacity and thickness)



