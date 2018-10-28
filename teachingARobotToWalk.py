"""Teaching a robot to walk using ai gym - midterm assignment
   Daniel Rosengarten
   daniel.rosengarten@gmail.com
   Process taken from Move37 Video - Augmented Random Search Tutorial - How to Train Robots to walk
"""

import gym
from gym import wrappers
class AugmentedRandomSearch():
    """Class for Augmented Random Search"""
    import normalizer
    from operator import add,sub
    def __init__(self, num_states, num_actions,options):
        import numpy as np
        self.weights = np.zeros((num_actions,num_states))
        self.options=options

    def should_record_step(self,step_number):
        """returns if a step should record based on RECORD_EVERY -
           callbackrequired by gym.wrappers.Monitor"""
        return step_number % self.options['RECORD_EVERY'] == 0

    def policy(self, state, addOrSubtractOperator,delta = None):
        """return adjusted (weights+noise) dot state"""
        if delta is None:
            return self.weights.dot(state)
        return (addOrSubtractOperator(self.weights,self.options['NOISE'])*delta).dot(state)

    def run_episode(self, env, normalizer, addOrSubtractOperator,delta=None, render = False):
        """Gets the total reward for an episode"""
        total_reward = 0
        state = env.reset()
        for episode_number in range(self.options['MAX_EPISODES']):
            if render: 
               env.render()
            normalizer.observe(state)
            state = normalizer.normalize(state)
            action = self.policy(state, addOrSubtractOperator,delta)
            state, reward, done, info = env.step(action)
            reward = max(min(reward, 1), -1)
            total_reward += reward
            if done: 
               break
        env.env.close()
        return total_reward
        
    def train(self, env, normalizer,RECORD_EVERY):
        """train using finite differences:
            1. Generate random noise, same shape as weights(theta)
            2. add/subtract noise to/from weights(theta) 
            3. collect rewards, getting rewards, determining the top 2
            4. adjust weights based on (best of positive delta reward- negative delta rewards) times learning rate (ALPHA),
            5. repeat """
        import numpy as np
        from operator import add,sub
        MAX_EPISODE_LEN = env.spec.timestep_limit or self.options['MAX_EPISODE_LEN']
        NUM_DELTAS=self.options['NUM_DELTAS']
        NUM_TOP_DELTAS=self.options['NUM_TOP_DELTAS']
        NUM_STEPS=self.options['NUM_STEPS']
        ALPHA=self.options['ALPHA']
        for step in range(NUM_STEPS):
            deltas = [np.random.randn(*self.weights.shape) for _ in range(NUM_DELTAS)]
            pos_delta_rewards = [0] * NUM_DELTAS
            neg_delta_rewards = [0] * NUM_DELTAS
            for i in range(NUM_DELTAS):
                pos_delta_rewards[i] = self.run_episode(env, normalizer,add,deltas[i]) 
                neg_delta_rewards[i] = self.run_episode(env, normalizer,sub,deltas[i]) 
            delta_rewards = np.array(pos_delta_rewards + neg_delta_rewards)
            rewards_and_deltas = [(pos_delta_rewards[i], neg_delta_rewards[i], deltas[i]) for i in range(NUM_DELTAS)]
            rewards_and_deltas.sort(key = lambda x : max(x[0:2]),reverse = True)
            rewards_and_deltas = rewards_and_deltas[:NUM_TOP_DELTAS]
            weight_adjustment_without_alpha = np.zeros(self.weights.shape)
            for positive_shift_rewards, negative_shift_rewards, delta in rewards_and_deltas:
                weight_adjustment_without_alpha += (positive_shift_rewards - negative_shift_rewards) * delta
            self.weights += ALPHA / (NUM_TOP_DELTAS * delta_rewards.std())*weight_adjustment_without_alpha
            reward = self.run_episode(env, normalizer,add,None)
            print("Step: #{} Reward: {}".format(step, reward))


def parse_config(filename):
    import configparser
    config = configparser.ConfigParser()
    config.read(filename)
    section='Environment'
    options={
            'ENV_NAME':config['Environment'].get('ENV_NAME'), 
            'MONITOR_DIR':config['Environment'].get('MONITOR_DIR'), 
            'RECORD_EVERY':config['Environment'].getint('RECORD_EVERY'), 
            'NUM_STEPS':config['AugmentedRandomSearch'].getint('NUM_STEPS'), 
            'MAX_EPISODES':config['AugmentedRandomSearch'].getint('MAX_EPISODES'), 
            'NUM_DELTAS':config['AugmentedRandomSearch'].getint('NUM_DELTAS'), 
            'NUM_TOP_DELTAS':config['AugmentedRandomSearch'].getint('NUM_TOP_DELTAS'), 
            'ALPHA':config['AugmentedRandomSearch'].getfloat('ALPHA'), 
            'NOISE':config['AugmentedRandomSearch'].getfloat('NOISE')
            }
    return options
    
def trainRobotToWalk():
   from normalizer import Normalizer
   options=parse_config('teachingARobotToWalk.config')
   env = gym.make(options['ENV_NAME']) 
   num_states,num_actions = env.observation_space.shape[0],env.action_space.shape[0]
   print ('options:{}'.format(options))
   agent = AugmentedRandomSearch(num_states, num_actions,options) 

   env = gym.wrappers.Monitor(env, options['MONITOR_DIR'], video_callable=agent.should_record_step ,force=True )
   normalizer = Normalizer(num_states)

   agent.train(env, normalizer, options['RECORD_EVERY'] ) # training for our agent

if __name__=='__main__':
   trainRobotToWalk()
