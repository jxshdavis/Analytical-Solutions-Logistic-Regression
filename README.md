# Analytic Logistic Regression


## 09.11.2020

TODO:
1. Figure out what journal I want to submit to.
2. Read one-step background questions.
3. Write down the equation for the update applied to our estimator.
4. Implement the one step update.
5. Make sure the user has the option to Remove rows below count k and delete homogenous rows.
6. Update Simulations.
7. Fix the hard coded "alternate" method where we either drop homogeneous rows or estimate them with $1-\frac{1}{2N_m}$ or $\frac{1}{2N_m}$.
8. Change the .transform_response() function to a .fit() named function.

## 09.17.2020 with McGuffey

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


## 09.18.2020 

### Updates
1. Read through LeCam Made Simple paper and took light notes on first section. I don't understand most of it right now.
2. Wrote down the one step update on paper. I still don't fully understand the assumptions needed to use it but I'm just treating it as a Newton's method step right now.
3. Got basic one step upworking working in code with synthetic examples.
4. Fixed issues causing MLE to not compute simulation (skip over cases where data is degenerate)


## 09.18.2020
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
1. Figured out Journal and began formatting abstract and intro.
2. Expanded the introduction for draft.

## 10.01.2024 McGuffey



### Udates:

1. Found the data set which illustrates the ability of analytic estimators.
2. Spread out histograms in simulation plot.
3. Fixed the hard coded "alternate" method... now a model parameter

   

