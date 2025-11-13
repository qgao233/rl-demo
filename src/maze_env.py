"""
4x4 迷宫环境
S: 起点 (0,0)
G: 终点 (3,3)
#: 障碍物
.: 可通行路径
"""
import numpy as np


class MazeEnv:
    def __init__(self):
        """
        初始化4x4迷宫环境
        迷宫布局:
        S . . .
        . # . #
        . . . .
        # . . G
        """
        self.grid_size = 4
        self.start_pos = (0, 0)
        self.goal_pos = (3, 3)
        
        # 障碍物位置
        self.obstacles = [(1, 1), (1, 3), (3, 0)]
        
        # 动作空间: 上、下、左、右
        self.actions = ['up', 'down', 'left', 'right']
        self.action_to_delta = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        self.current_pos = self.start_pos
        
    def reset(self):
        """重置环境到起点"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        """
        执行动作
        返回: (next_state, reward, done)
        """
        delta = self.action_to_delta[action]
        next_pos = (self.current_pos[0] + delta[0], 
                    self.current_pos[1] + delta[1])
        
        # 检查是否越界
        if not self._is_valid_position(next_pos):
            # 撞墙或越界，停在原地，给予负奖励
            return self.current_pos, -1, False
        
        # 检查是否是障碍物
        if next_pos in self.obstacles:
            # 撞到障碍物，停在原地，给予负奖励
            return self.current_pos, -1, False
        
        # 移动到新位置
        self.current_pos = next_pos
        
        # 检查是否到达终点
        if self.current_pos == self.goal_pos:
            return self.current_pos, 100, True  # 到达终点，给予大奖励
        
        # 正常移动，给予小负奖励（鼓励尽快到达终点）
        return self.current_pos, -0.1, False
    
    def _is_valid_position(self, pos):
        """检查位置是否在网格内"""
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def get_all_states(self):
        """获取所有有效状态"""
        states = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in self.obstacles:
                    states.append((i, j))
        return states
    
    def render(self, agent_pos=None):
        """可视化迷宫"""
        if agent_pos is None:
            agent_pos = self.current_pos
            
        print("\n迷宫状态:")
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                if (i, j) == agent_pos:
                    row_str += "A "  # Agent
                elif (i, j) == self.goal_pos:
                    row_str += "G "  # Goal
                elif (i, j) == self.start_pos:
                    row_str += "S "  # Start
                elif (i, j) in self.obstacles:
                    row_str += "# "  # Obstacle
                else:
                    row_str += ". "  # Empty
            print(row_str)
        print()
    
    def render_path(self, path):
        """可视化路径"""
        print("\n最优路径:")
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                if (i, j) in path:
                    if (i, j) == self.start_pos:
                        row_str += "S "
                    elif (i, j) == self.goal_pos:
                        row_str += "G "
                    else:
                        row_str += "* "  # Path
                elif (i, j) in self.obstacles:
                    row_str += "# "
                else:
                    row_str += ". "
            print(row_str)
        print()

