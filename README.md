# DiaMetricsLearner
Providing personalized insight to your Diabetes using data science.

# Project 1 - Fitting Sigmoids 
We fit sigmoids to Dexcom data to extract carbs-to-bg and insulin-to-bg curves. Basically, you'll know the expected bg for the next 3/4 hours before it happens. (uses simulated annealing and sigmoids to solve. each day is an episode and the average of the curves will be taken)

# Project 2 - Q-Learning
The next project will be choosing optimal actions to have really good bg control. This will start with q-learning and may also involve the simulated annealing results from project 1.

### BGEnvironment-v0
   *Description*
   
   Simplified environment for modeling the blood glucose response of the human body
   at the beginning of the day with one meal and one insulin injection. Episodes 
   last 3 hours or 36 time-steps.
      
   State Fields:  
   * blood glucose     (in units of 10 mg/dL) 
   * insulin taken     (in units of 0.5 increments)
   * carbs eaten       (in units of 5 grams)
   * time since meal   (in units of 5 minute intervals)
           
   Actions: 
   * Take Insulin      (in units of 0.5 increments)
   * Eat Carbohydrates (in units of 5 grams)
            
   Scoring: Sum(|episode[t] - 140|) for all t
