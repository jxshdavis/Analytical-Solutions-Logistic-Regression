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



