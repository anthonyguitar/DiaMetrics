import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

class Sigmoid:
    """Class that applies sigmoid functions to discrete data."""
    @staticmethod
    def logistic(scale, shift, stretch, t):
        """f(a, b, c, t) -> a / (1 + e^(-(t-b)/c))."""
        return scale / (1 + np.power(np.e, -1.0*(t - shift )/ stretch))

class SigmoidFitProblem:
    """A model using sigmoids to model glucose response.

    Class variables:
    start_time -- When the episode started (datetime).
    end_time   -- When the episode ended (datetime).
    bg_initial -- Blood glucose at beginning of episode.
    meals      -- List of meals consumed during episode. Each meal is a
                  tuple (meal time, number of carbs eaten).
    injections -- List of injections taken during episode. Each injection is a
                  tuple (injection time, number of units taken).
    time_data  -- The time axis, each tick is 5 minute interval.
    bg_data    -- List of blood sugar readings. There is one reading per 5
                  minutes during the episode.
    S          -- State containing the parameters of the best fit for
                  sigmoids.
    """


    def __init__(self, start_time, end_time, bg_initial):
        """Initialize class variables.

        Keyword arguments:
        start_time -- Time in which the episode started (datetime).
        end_time   -- Time in which the episode ends (datetime).
        bg_initial -- The initial blood sugar value.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.bg_initial = bg_initial
        self.meals = []
        self.injections = []
        
        time_axis = (end_time - start_time).total_seconds() / 300.0
        self.time_data = np.arange(0, time_axis, 1.0)
        self.bg_data = np.zeros(self.time_data.shape)
        self.attr_factors = [0.025, 0.5, 0.5, 0.5, 0.025, 0.05, 0.05]

    def add_meal(self, meal_time, carbs):
        """Add a meal to the problem.

        Keyword arguments:
        meal_time -- Time in which meal was consumed (datetime).
        carbs     -- Number of carbs consumed during meal.
        """
        self.meals.append((meal_time, carbs))

    def add_injection(self, injection_time, insulin_amount):
        """Add an insulin injection to the problem.

        Keyword arguments:
        injection_time -- Time in which the injection was taken (datetime).
        insulin_amount -- Number of units taken.
        """
        self.injections.append((injection_time, insulin_amount))

    def set_target_data(self, data):
        """Sets the target data that will be used to solve for params.

        Keyword arguments:
        data -- input data.
        """
        if not data.shape == self.bg_data.shape:
            print("Error, the data must be the same dimensions as episode.")
        else:
            self.bg_data = data

    def init_paramters(self):
        """Initialize state to have reasonable parameters."""
        carb_bg_ratio = 5.0
        time_to_breakdown = 45.0
        insulin_bg_ratio = 50.0
        time_to_peak = 45.0
        basal_rate = 0.0
        digestion_speed = 1.0
        activation_speed = 1.0

        # set state to initial
        self.S = [self.carb_bg_ratio, self.time_to_breakdown,
                  self.insulin_bg_ratio, self.time_to_peak,
                  self.basal_rate, self.digestion_speed,
                  self.activation_speed]
                  
    def get_random_neighbor(self):
        """Get a new state that is very close to current."""
        r_mask = np.random.randint(0, 2, 7)
        r_exp = np.random.randint(0, 2, 7)
        changes = r_mask*self.attr_factors*np.power(-1.0, r_exp)
        Sp = self.S + changes
        basal_C = Sp[4]
        Sp = np.maximum(Sp, 0.0)
        Sp[4] = basal_C
        return Sp
        
    def simulate_data(self, S):
        """Simulate blood sugar response based on input parameters.

        Keyword arguments:
        S -- State consisting of the following:
        carb_bg_ratio     -- Blood sugar rise per gram of carbohydrate.
        digestion_speed   -- The time it takes for half of carbs to be
                             digested.
        time_to_breakdown -- How long it takes for carbs to digest once
                             they have started breaking down.
        insulin_bg_ratio  -- Blood sugar drop per unit of insulin.
        time_to_peak      -- The time it takes for half of insulin to be
                             activated.
        activation_speed  -- How long it takes for insulin to finish
                             working after it has begun.
        basal_delta       -- Blood glucose change per 5 minutes due to
                             basal insulin.

        Returns:
        list of simulated bg values.
        """
        
        # extract parameters from state to make equations clear
        (carb_bg_ratio, time_to_breakdown, insulin_bg_ratio, 
         time_to_peak, basal_delta, digestion_speed,
         activation_speed) = S
        

        # initialize blood glucose data
        simulated_data = np.full(self.bg_data.shape,
                                 self.bg_initial, dtype=float)

        # simulate bg effect due to basal
        simulated_data += self.time_data*basal_delta

        # simulate bg effect due to meals
        for meal in self.meals:

            # starting time point
            time_delta = (meal[0] - self.start_time).total_seconds()
            discrete_start_time = int(time_delta / 300.0)
            discrete_time_to_breakdown = time_to_breakdown / 5.0
            bg_rise = meal[1]*carb_bg_ratio
            bg_shift = discrete_start_time + discrete_time_to_breakdown
            meal_effect = Sigmoid.logistic(
                              bg_rise, bg_shift, digestion_speed,
                              self.time_data[discrete_start_time:])
            simulated_data += np.concatenate((np.zeros(discrete_start_time),
                                              meal_effect))

        # simulate bg effect due to injections
        for injection in self.injections:
            time_delta = (injection[0] - self.start_time).total_seconds()
            discrete_start_time = int(time_delta / 300.0)
            discrete_time_to_peak = time_to_peak / 5.0
            insulin_rise = injection[1]*insulin_bg_ratio
            insulin_shift = discrete_start_time + discrete_time_to_peak
            insulin_effect = Sigmoid.logistic(
                                 insulin_rise,
                                 insulin_shift,
                                 activation_speed,
                                 self.time_data[discrete_start_time:])
            simulated_data -= np.concatenate((np.zeros(discrete_start_time),
                                              insulin_effect))

        # return the simulated data
        return simulated_data


def simulated_annealing(problem, iterations=150000, T_init=1000,
                        T_decay=0.00015):
    """Find best sigmoid parameters via simulated annealing.

    Keyword arguments:
    domain     -- The problem in which we are to solve using simulated
                  annealing. (default SigmoidFitProblem)
    iterations -- The number of iterations to run simulated annealing
                  (default 150,000)
    T_init     -- The initial temperature (default 1,000)
    T_decay    -- The exponential factor used to anneal the temperature
                  (default 0.00015)
    """
    # initialize parameters and get initial state
    problem.init_paramters()

    # get the response from the initial state
    simulated = problem.simulate_data(problem.S)

    # calculate initial error of state compared to target
    error = np.average(np.square(problem.bg_data - simulated))

    # generate random numbers for choosing action up-front
    rnums = np.random.random(iterations)

    for iteration in range(iterations):

        # anneal temperature
        T = T_init*np.exp(-1.0*T_decay*iteration)

        # get a random next state (neighbor)
        Sp = problem.get_random_neighbor()

        # get the response from the new state
        simulated = problem.simulate_data(Sp)

        # calculate error of new data compared to target
        new_error = np.average(np.square(problem.bg_data - simulated))

        # choose neighbor with prob 1 - exp(delta(E)/kT))
        deltaE = new_error - error
        P = np.exp(-1.0*(deltaE) / T)
        if deltaE <= 0 or rnums[iteration] < P:
            problem.S = Sp
            error = new_error

if __name__ == '__main__':

    # Setup sigmoid fit problem
    problem = SigmoidFitProblem(start_time = dt.datetime(2020, 1, 1, 8, 0, 0),
                                end_time = dt.datetime(2020, 1, 1, 15, 0, 0),
                                bg_initial = 100)

    # add breakfast
    problem.add_meal(meal_time = dt.datetime(2020, 1, 1, 8, 0, 0),
                     carbs = 28.0)

    # take insulin for breakfast meal
    problem.add_injection(injection_time = dt.datetime(2020, 1, 1, 8, 5, 0),
                          insulin_amount = 1.5)


    # add lunch
    problem.add_meal(meal_time = dt.datetime(2020, 1, 1, 10, 0, 0),
                     carbs = 8.0)

    # provide target data to problem
    carb_bg_ratio = 6.0
    digestion_speed = 1.0
    time_to_breakdown = 45.0
    insulin_bg_ratio = 80.0
    time_to_peak = 45.
    activation_speed = 1.0
    basal_rate = -0.5
    S_fake = [carb_bg_ratio, time_to_breakdown, insulin_bg_ratio, time_to_peak,
              basal_rate, digestion_speed, activation_speed]
    fake_data = problem.simulate_data(S_fake)

    # set the actual data equal to the simulated data
    problem.set_target_data(data = fake_data)

    # run simulated annealing to find most likely parameters
    simulated_annealing(problem)

    # view final results
    fitted_data = problem.simulate_data(problem.S)

    plt.plot(problem.time_data, fake_data)
    plt.plot(problem.time_data, fitted_data)
    plt.show()
