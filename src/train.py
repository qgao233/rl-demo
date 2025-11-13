"""
训练Q-Learning智能体在迷宫中找到最优路径
"""
import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from q_learning import QLearningAgent


def train(episodes=500, max_steps=100):
    """
    训练Q-Learning智能体
    
    参数:
    - episodes: 训练轮数
    - max_steps: 每轮最大步数
    """
    # 创建环境和智能体
    env = MazeEnv()
    agent = QLearningAgent(
        actions=env.actions,
        learning_rate=0.1,    # α
        discount_factor=0.9,  # γ
        epsilon=0.1           # ε
    )
    
    # 记录训练过程
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    print("开始训练Q-Learning智能体...")
    print(f"参数设置: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon}")
    
    # 显示初始迷宫
    env.render()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # 行动策略: ε-greedy选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # Q-Learning更新（异策略）
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                success_count += 1
                break
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # 每100轮打印一次进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            success_rate = success_count / (episode + 1) * 100
            print(f"轮次 {episode + 1}/{episodes} - "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"平均步数: {avg_steps:.2f}, "
                  f"成功率: {success_rate:.1f}%")
    
    print(f"\n训练完成! 总成功率: {success_count/episodes*100:.1f}%")
    
    return env, agent, episode_rewards, episode_steps


def test_agent(env, agent):
    """测试训练好的智能体"""
    print("\n" + "="*50)
    print("测试训练好的智能体（使用贪婪策略）")
    print("="*50)
    
    state = env.reset()
    path = [state]
    total_reward = 0
    steps = 0
    max_steps = 20
    
    print(f"\n起点: {state}")
    
    while steps < max_steps:
        # 使用贪婪策略（不探索）
        action = agent.get_policy(state)
        next_state, reward, done = env.step(action)
        
        path.append(next_state)
        total_reward += reward
        steps += 1
        
        print(f"步骤 {steps}: {state} --{action}--> {next_state}, 奖励: {reward:.2f}")
        
        state = next_state
        
        if done:
            print(f"\n成功到达终点! 总步数: {steps}, 总奖励: {total_reward:.2f}")
            break
    else:
        print(f"\n未能在{max_steps}步内到达终点")
    
    # 可视化路径
    env.render_path(path)
    
    return path, total_reward, steps


def plot_training_results(episode_rewards, episode_steps):
    """绘制训练结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制奖励曲线
    ax1.plot(episode_rewards, alpha=0.3, color='blue')
    # 绘制移动平均
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, 
                                mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), 
                moving_avg, 
                color='red', 
                linewidth=2, 
                label=f'{window}轮移动平均')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('总奖励')
    ax1.set_title('Q-Learning训练过程 - 奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制步数曲线
    ax2.plot(episode_steps, alpha=0.3, color='green')
    if len(episode_steps) >= window:
        moving_avg = np.convolve(episode_steps, 
                                np.ones(window)/window, 
                                mode='valid')
        ax2.plot(range(window-1, len(episode_steps)), 
                moving_avg, 
                color='red', 
                linewidth=2, 
                label=f'{window}轮移动平均')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('步数')
    ax2.set_title('Q-Learning训练过程 - 步数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("\n训练结果图已保存为 'training_results.png'")
    plt.show()


if __name__ == "__main__":
    # 训练智能体
    env, agent, episode_rewards, episode_steps = train(episodes=500, max_steps=100)
    
    # 打印部分Q表
    print("\n部分Q表（前5个状态）:")
    states = sorted(agent.q_table.keys())[:5]
    for state in states:
        print(f"\n状态 {state}:")
        for action in env.actions:
            print(f"  {action}: {agent.q_table[state][action]:.3f}")
    
    # 测试智能体
    path, total_reward, steps = test_agent(env, agent)
    
    # 绘制训练结果
    plot_training_results(episode_rewards, episode_steps)

