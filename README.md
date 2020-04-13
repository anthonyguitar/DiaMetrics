# DiaMetricsLearner
Modeling blood glucose response as a Markov Decision Process.

# Environments
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
