"""
Q-Learning 算法实现
更新公式: Q(S,A)←Q(S,A)+α[R+γ max Q(S',a')−Q(S,A)]
                              a'
"""
import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        初始化Q-Learning智能体
        
        参数:
        - actions: 动作空间列表
        - learning_rate (α): 学习率
        - discount_factor (γ): 折扣因子
        - epsilon (ε): ε-greedy策略中的探索概率
        """
        self.actions = actions
        self.alpha = learning_rate  # 学习率 α
        self.gamma = discount_factor  # 折扣因子 γ
        self.epsilon = epsilon  # 探索概率 ε
        
        # Q表: Q(state, action) -> value
        # 使用defaultdict自动初始化为0
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def choose_action(self, state, use_epsilon=True):
        """
        行动策略（Behavior Policy）: ε-greedy
        
        - 以概率 ε 随机选择动作（探索）
        - 以概率 1-ε 选择Q值最大的动作（利用）
        """
        if use_epsilon and random.random() < self.epsilon:
            # 探索: 随机选择动作
            return random.choice(self.actions)
        else:
            # 利用: 选择Q值最大的动作
            q_values = [self.q_table[state][action] for action in self.actions]
            max_q = max(q_values)
            
            # 如果有多个动作具有相同的最大Q值，随机选择一个
            max_actions = [action for action, q in zip(self.actions, q_values) 
                          if q == max_q]
            return random.choice(max_actions)
    
    def get_max_q_value(self, state):
        """
        评估策略（Target Policy）: 贪婪策略
        
        获取状态state下所有动作的最大Q值: max Q(S', a')
                                          a'
        """
        q_values = [self.q_table[state][action] for action in self.actions]
        return max(q_values) if q_values else 0.0
    
    def update(self, state, action, reward, next_state):
        """
        Q-Learning更新规则（异策略: off-policy）
        
        Q(S,A) ← Q(S,A) + α[R + γ max Q(S',a') - Q(S,A)]
                                    a'
        
        参数:
        - state (S): 当前状态
        - action (a): 当前动作
        - reward (R): 获得的奖励
        - next_state (S'): 下一个状态
        """
        current_q = self.q_table[state][action]
        
        # 评估策略: 使用贪婪策略获取下一状态的最大Q值
        max_next_q = self.get_max_q_value(next_state)
        
        # Q-Learning更新公式
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
    
    def get_policy(self, state):
        """获取当前状态下的最优动作（不使用ε-greedy）"""
        return self.choose_action(state, use_epsilon=False)
    
    def print_q_table(self):
        """打印Q表"""
        print("\nQ表:")
        for state in sorted(self.q_table.keys()):
            print(f"状态 {state}:")
            for action in self.actions:
                print(f"  {action}: {self.q_table[state][action]:.2f}")

