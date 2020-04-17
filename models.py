import matplotlib.pyplot as plt
import numpy as np
import datetime

def logistic_function(scale, shift, stretch, t):
   """scale: Final value of function.
      shift: How far function is shifted to the right.
      t    : Input time series.
   """
   return(scale/(1 + np.power(np.e, -1.0*(t-shift)/stretch))) 
   
def eat_meal(start_time, meal_time, carbs, carb_bg_ratio, time_to_breakdown, digestion_speed, t):
   # starting time point
   discrete_start_time = int((meal_time - start_time).total_seconds()/300.0)
   discrete_time_to_breakdown = time_to_breakdown/5.0
   meal_effect = logistic_function(carbs*carb_bg_ratio, 
                                   discrete_start_time+discrete_time_to_breakdown, 
                                   digestion_speed,
                                   t[discrete_start_time:])  
   return np.concatenate((np.zeros(discrete_start_time), meal_effect))
   
def take_insulin(start_time, injection_time, insulin, insulin_bg_ratio, time_to_peak, activation_speed, t):
   # starting time point
   discrete_start_time = int((injection_time - start_time).total_seconds()/300.0)
   discrete_time_to_peak = time_to_peak/5.0
   insulin_effect = logistic_function(insulin*insulin_bg_ratio, 
                                      discrete_start_time+discrete_time_to_peak,
                                      activation_speed,
                                      t[discrete_start_time:])  
   return np.concatenate((np.zeros(discrete_start_time), insulin_effect))
   
def basal_effect(change_per_discrete, t):
   return t*change_per_discrete
   
def gen_neighbors(S, i, jump):
   
   if i==4:
      return [S]
   
   next_neighbors = []

   for change in [-1.0*jump, 0, 1.0*jump]:
      Sp = list(S)
      if i == 0:
         Sp[i] = max(Sp[i]+change/20.0, 0)
      else:
         Sp[i] = max(Sp[i]+change, 0)
      next_neighbors += gen_neighbors(tuple(Sp), i+1, jump)
   return next_neighbors
   
   
# Let's say that we have 12 hours to work with
# Breakfast is at 8:00 AM, We take insulin at 8:05 AM
# Then nothing happens the rest of the day
start_time = datetime.datetime(2020, 1, 1, 8, 0, 0)
end_time   = datetime.datetime(2020, 1, 1, 15, 0, 0)
time_axis  = (end_time - start_time).total_seconds()/300

# create time axis
t = np.arange(0, time_axis, 1.0)

# specify starting BG of the day
starting_bg = 100.0

# add basal effect
net_basal_effect = -0.5 # per discrete point (5 minutes)
basal = basal_effect(net_basal_effect, t)

# add meal that happens at 8:00
breakfast_time = datetime.datetime(2020, 1, 1, 8, 0, 0)
carbs = 28.0
carb_bg_ratio = 6.0 # increase per carb
carb_digestion_speed = 1.0
time_to_breakdown = 45.0 # in minutes
breakfast = eat_meal(start_time, breakfast_time, carbs, carb_bg_ratio, 
                     time_to_breakdown, carb_digestion_speed, t)

# take insulin
injection_time = datetime.datetime(2020, 1, 1, 8, 0, 0)
insulin = 1.0
insulin_bg_ratio = 100.0 # drop per unit
time_to_peak = 60.0 # in minutes
insulin_activation_speed = 1.0
injection = take_insulin(start_time, injection_time, insulin, insulin_bg_ratio, 
                         time_to_peak, insulin_activation_speed, t)
                         
                        
# add meal that happens at 12:00
lunch_time = datetime.datetime(2020, 1, 1, 11, 0, 0)
lunch_carbs = 9.0
carb_bg_ratio = 6.0 # increase per carb
time_to_breakdown = 45.0 # in minutes
lunch = eat_meal(start_time, lunch_time, lunch_carbs, carb_bg_ratio, 
                 time_to_breakdown, carb_digestion_speed, t)

# take insulin for lunch
injection_time_lunch = datetime.datetime(2020, 1, 1, 10, 30, 0)
insulin_lunch = 0.5
injection_lunch = take_insulin(start_time, injection_time_lunch, insulin_lunch, insulin_bg_ratio, 
                         time_to_peak, insulin_activation_speed, t)

# add true observation to graph
observed_event = starting_bg + basal + breakfast + lunch - injection - injection_lunch

# now pretend like we don't have insulin ratio
carb_bg_ratio     = 5.0
time_to_breakdown = 45.0
insulin_bg_ratio  = 50.0
time_to_peak      = 45.0
basal_rate        = 0.0
digestion_speed   = 1.0
activation_speed  = 1.0

# run simulated annealing
# State = [carb-bg-ratio, time-to-breakdown, insulin-bg-ratio, time-to-peak]
S = [carb_bg_ratio, time_to_breakdown, insulin_bg_ratio, time_to_peak, basal_rate, digestion_speed, activation_speed]

# get init error
basal     = basal_effect(S[4], t)
breakfast = eat_meal(start_time, breakfast_time, carbs, S[0], S[1], S[5], t)
lunch     = eat_meal(start_time, lunch_time, lunch_carbs, S[0], S[1], S[5], t)
injection = take_insulin(start_time, injection_time, insulin, S[2], S[3], S[6], t)
injection_lunch = take_insulin(start_time, injection_time_lunch, insulin_lunch, S[2], S[3], S[6], t)
estimate  = starting_bg + basal + breakfast +lunch - injection - injection_lunch
error     = np.average(np.square(observed_event - estimate))

# hyper-parameters
KMAX = 100000
T_init = 1000
T_decay = 0.0002
errors = []
rnums = np.random.random(KMAX)
attr_factors = [0.025, 0.5, 0.5, 0.5, 0.025, 0.05, 0.05]

for k in range(KMAX):
   
   # anneal temperature
   T = T_init*np.exp(-1.0*T_decay*k)
   
   # find possible neighbors
   
   # choose random neighbor
   r_mask  = np.random.randint(0, 2, 7)
   r_exp   = np.random.randint(0, 2, 7)
   changes = r_mask*attr_factors*np.power(-1.0, r_exp)
   Sp      = S + changes
   basal_C = Sp[4]
   Sp      = np.maximum(Sp, 0.0)
   Sp[4]   = basal_C
   
   
   # implement the neighbor      
   basal     = basal_effect(Sp[4], t)
   breakfast = eat_meal(start_time, breakfast_time, carbs, Sp[0], Sp[1], Sp[5], t)
   lunch     = eat_meal(start_time, lunch_time, lunch_carbs, Sp[0], Sp[1], Sp[5], t)
   injection = take_insulin(start_time, injection_time, insulin, Sp[2], Sp[3], Sp[6], t)
   injection_lunch = take_insulin(start_time, injection_time_lunch, insulin_lunch, Sp[2], Sp[3], Sp[6], t)
   estimate  = starting_bg + basal + breakfast + lunch - injection - injection_lunch
   new_error = np.average(np.square(observed_event - estimate))
   
   # choose neighbor with prob 1 - exp(delta(E)/kT))
   deltaE = new_error - error
   P = np.exp(-(deltaE)/T)

   if deltaE <= 0 or rnums[k] < P:
      S = Sp
      error = new_error
   #if k % 25000 == 0:
   #   plt.plot(t, estimate, label='SA {0}'.format(k/25000))
  
   errors.append(error)

# print best state
print(S)    

# get best error
basal     = basal_effect(S[4], t)
breakfast = eat_meal(start_time, breakfast_time, carbs, S[0], S[1], S[5], t)
lunch     = eat_meal(start_time, lunch_time, lunch_carbs, S[0], S[1], S[5], t)
injection = take_insulin(start_time, injection_time, insulin, S[2], S[3], S[6], t)
injection_lunch = take_insulin(start_time, injection_time_lunch, insulin_lunch, S[2], S[3], S[6], t)
estimate  = starting_bg + basal + breakfast + lunch - injection - injection_lunch

plt.plot(t, observed_event, label='Truth')
plt.plot(t, estimate, label='Final SA')
plt.legend()
plt.show()


plt.plot(errors)
plt.show()
