import math
import numpy as np

class DiaMetricsLearner:
   def __init__(self):
      self.__version__ = '0.0.1'
      
class BGEnvironment_v0:
   """
   Simplified environment for modeling the blood glucose response of the human body
   at the beginning of the day with one meal and one insulin injection. Episodes 
   last 3 hours or 36 time-steps.
      
   State: <blood glucose,      (in units of 10 mg/dL) 
           insulin taken,      (in units of 0.5 increments)
           carbs eaten,        (in units of 5 grams)
           time since meal>    (in units of 5 minute intervals)
           
   Action: <Take Insulin,      (in units of 0.5 increments)
            Eat Carbohydrates> (in units of 5 grams)
            
   Score(episode): Sum(|episode[t] - 140|) for all t
   """
              
   def __init__(self):
      
      # Environment version number
      self.__version__ = 'v0'
      
      # define the limits of our environment
      self.min_carbs        = 0    # in mg/dL
      self.max_carbs        = 60   # in mg/dL
      self.min_insulin      = 0    # in units
      self.max_insulin      = 3    # in units
      self.min_bg           = 60   # in mg/dL
      self.max_bg           = 250  # in mg/dL
      self.episode_length   = 60*3 # in minutes
      self.target_bg        = 140  # in mg/dL
      
      # needed to reduce state and action space
      self.bg_divisor        = 10
      self.carbs_divisor     = 5
      self.insulin_divisor   = 0.5
      self.time_divisor      = 5
      
      # For indexing into a state
      self.bg_index          = 0
      self.insulin_index     = 1
      self.carbs_index       = 2
      self.time_index        = 3
      
      # calculate the action-space dimensions using the defined limits
      self.n_carb_actions = math.ceil(self.max_carbs - self.min_carbs)/self.carbs_divisor + 1
      self.n_insulin_actions = int((self.max_insulin - self.min_insulin)/self.insulin_divosor + 1)
      self.n_actions = (self.n_insulin_actions, self.n_carb_actions)
      
      # calculate the state-space dimension using the defined limits
      self.n_bg_states = math.ceil(self.max_bg - self.min_bg)/self.bg_divisor + 1
      self.n_timesteps = int(self.episode_length/self.time_divisor)

      # initialize Q-matrix that will learn during q-learning
      self.Q = np.zeros((self.n_bg_states, self.n_insulin_actions, self.n_carb_actions, 
                         self.n_timesteps))      

   def score(S):
      """Return the value of a single BG reading."""
      return np.abs(S[self.bg_index]*self.bg_divisor + self.min_bg - self.target_bg)

# For troubleshooting
if __name__ == '__main__':
   env = BGEnvironment_v0()
   