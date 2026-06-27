# D1H 双轮足机器人单腿稳定站立技能 — 实现计划

## 1. 目标概述

在现有 d1h_flat 行走技能的基础上，新增一个 **单腿稳定站立（one-leg standing）** 技能。机器人需要：
- 抬起一条腿，仅靠另一条腿的轮子维持平衡
- 保持躯干姿态稳定（不倾倒、不漂移）
- 站立腿的轮子提供平衡力矩（类似倒立摆平衡）

## 2. 核心挑战分析

| 挑战 | 说明 |
|------|------|
| 倒立摆平衡 | 单腿站立时机器人本质是一个倒立摆，需要轮子持续提供平衡力矩 |
| 质心偏移 | 抬腿后质心偏向站立侧，需要髋关节补偿 |
| 动作空间 | 抬起腿的关节需要收敛到固定姿态，站立腿的轮子需要主动平衡 |
| 终止条件 | 倾倒/触地检测需要更严格 |

## 3. 需要新建/修改的文件

### 3.1 新建文件

| 文件 | 说明 |
|------|------|
| `configs/d1h/d1h_one_leg_stand_config.py` | 单腿站立的环境类 + 配置类 |

### 3.2 修改文件

| 文件 | 说明 |
|------|------|
| `configs/d1h/__init__.py` | 注册新任务 `d1h_one_leg_stand` / `d1h_one_leg_stand_play` |

## 4. 详细修改方案

---

### 4.1 新建 `d1h_one_leg_stand_config.py`

#### 4.1.1 环境类 `D1HOneLegStand(LeggedRobot)`

**需要覆写的方法：**

##### (a) `_init_buffers(self)`
- 调用 `super()._init_buffers()`
- 定义关节索引：`hip_joint_indices=[0,4]`, `thigh_joint_indices=[1,5]`, `calf_joint_indices=[2,6]`, `foot_joint_indices=[3,7]`
- 新增：`stand_leg_indices`（站立腿的 hip/thigh/calf/foot 索引）和 `lift_leg_indices`（抬起腿的索引）
- 新增：`lift_leg_contact_buf` 用于追踪抬起腿是否触地

##### (b) `_reset_root_states(self, env_ids)`
- **关键修改**：重置时随机选择站立腿（左腿或右腿），每 50% 概率
- 初始高度设为单腿站立高度（约 0.45m，与双腿站立一致）
- 初始姿态：**减小随机范围**，roll/pitch 随机范围从 `[-π, π]` 缩小到 `[-0.3, 0.3]`，yaw 仍可全范围随机
  - 原因：单腿站立初始姿态太离谱会直接倒下，不利于学习
- 初始速度：减小随机范围至 `[-0.1, 0.1]`

##### (c) `_reset_dofs(self, env_ids)` （需覆写）
- 站立腿：使用默认关节角度（hip=0, thigh=0.8, calf=-1.5）
- 抬起腿：设置为收起姿态（hip=0, thigh≈1.2, calf≈-2.5，即大腿前抬、小腿收紧）
- 添加小范围随机扰动 `±0.1`

##### (d) `check_termination(self)`
- 保留原有终止条件（base触地、超时、高度<0）
- **新增终止条件**：
  - 抬起腿的脚触地 → 终止（训练初期强制要求抬腿）
  - 躯干倾斜过大（projected_gravity_z < 0.5，即偏离竖直超过约60°）→ 终止
  - 站立腿的大腿/小腿触地 → 终止

##### (e) `step(self, actions)`
- 与 `D1HFlat.step()` 基本一致
- **关键**：抬起腿的轮子关节（foot_joint）也需要 `dof_pos = 0`（保持自由旋转或锁定均可，建议锁定以减少干扰）

##### (f) `_compute_torques(self, actions)`
- 与 `D1HFlat` 一致
- 可选：对抬起腿的关节增加额外阻尼，帮助其收敛到目标姿态

##### (g) `compute_observations(self)` — 不需要覆写
- 沿用基类的观测空间（33维本体感知 + 187维地形扫描 + 历史 + 特权信息）
- 地形扫描在平地上全为0，不影响

##### (h) 新增/修改的奖励函数

| 奖励名 | 类型 | 权重 | 说明 |
|--------|------|------|------|
| `orientation` | 惩罚 | **-20.0** | 姿态保持，比行走时更强（原-10） |
| `lin_vel_z` | 惩罚 | **-2.0** | 垂直速度惩罚增强（原-1） |
| `ang_vel_xy` | 惩罚 | **-0.5** | roll/pitch角速度惩罚增强（原-0.05） |
| `base_height` | 惩罚 | **-15.0** | 高度偏离惩罚增强（原-10） |
| `upward` | 奖励 | **5.0** | 竖直朝上奖励增强（原2） |
| `stand_foot_contact` | 奖励 | **10.0** | **新增**：站立腿脚轮持续触地奖励 |
| `lift_foot_no_contact` | 奖励 | **8.0** | **新增**：抬起腿脚轮不触地奖励 |
| `lift_leg_target` | 奖励 | **5.0** | **新增**：抬起腿关节角度接近目标姿态奖励 |
| `balance_center` | 奖励 | **3.0** | **新增**：质心投影在站立脚上方奖励 |
| `stand_wheel_still` | 惩罚 | **-0.1** | **新增**：站立腿轮子不必要的旋转惩罚（当命令为零时） |
| `action_rate` | 惩罚 | **-0.02** | 动作平滑性惩罚增强（原-0.01） |
| `collision` | 惩罚 | **-10.0** | 碰撞惩罚增强（原-2） |
| `collision_head` | 惩罚 | **-20.0** | 躯干触地惩罚增强（原-5） |
| `tracking_lin_vel_x` | 奖励 | **5.0** | 大幅降低（原30，站立时不需要快速移动） |
| `tracking_lin_vel_y` | 奖励 | **0.0** | 关闭（站立时不需要横向移动） |
| `tracking_ang_vel` | 奖励 | **5.0** | 降低（原15，站立时缓慢转向即可） |
| `feet_air_time` | 奖励 | **0.0** | 关闭（不需要步态） |
| `keep_still` | 惩罚 | **-2.0** | 增强（原-0.5，站立时更需静止） |
| `body_pos_to_feet_x` | 奖励 | **2.0** | 增强（原0.5，质心需对准站立脚） |
| `body_feet_distance_y` | 惩罚 | **0.0** | 关闭（单腿时双脚y距离无意义） |
| `body_symmetry_y` | 奖励 | **0.0** | 关闭（单腿时无需对称） |
| `body_symmetry_z` | 奖励 | **0.0** | 关闭（单腿时无需对称） |
| `termination` | 惩罚 | **-200.0** | 增强（原-100） |
| `powers` | 惩罚 | **-5e-5** | 增强能耗惩罚（原-2e-5） |

**新增奖励函数实现：**

```python
def _reward_stand_foot_contact(self):
    """站立腿脚轮持续触地奖励"""
    stand_foot_force = self.contact_forces[:, self.stand_foot_index, 2]
    return (stand_foot_force > 1.0).float()

def _reward_lift_foot_no_contact(self):
    """抬起腿脚轮不触地奖励"""
    lift_foot_force = torch.norm(self.contact_forces[:, self.lift_foot_index, :], dim=-1)
    return (lift_foot_force < 1.0).float()

def _reward_lift_leg_target(self):
    """抬起腿关节角度接近目标姿态奖励"""
    lift_hip_err = torch.square(self.dof_pos[:, self.lift_hip_idx] - self.lift_leg_target_hip)
    lift_thigh_err = torch.square(self.dof_pos[:, self.lift_thigh_idx] - self.lift_leg_target_thigh)
    lift_calf_err = torch.square(self.dof_pos[:, self.lift_calf_idx] - self.lift_leg_target_calf)
    return torch.exp(-(lift_hip_err + lift_thigh_err + lift_calf_err) / 0.5)

def _reward_balance_center(self):
    """质心投影在站立脚上方奖励"""
    # 站立脚在base坐标系下的x偏移
    foot_pos_base = quat_rotate_inverse(self.base_quat,
        self.feet_pos[:, self.stand_foot_idx, :] - self.root_states[:, :3])
    x_offset = torch.abs(foot_pos_base[:, 0])
    y_offset = torch.abs(foot_pos_base[:, 1])
    return torch.exp(-(x_offset + y_offset) / 0.1)
```

##### (i) Cost 函数

| Cost 名 | 权重 | 说明 |
|---------|------|------|
| `pos_limit` | 0.3 | 关节位置限制 |
| `torque_limit` | 0.3 | 力矩限制 |
| `dof_vel_limits` | 0.3 | 速度限制 |
| `default_joint` | 0.3 | 站立腿偏离默认角度惩罚 |

---

#### 4.1.2 配置类 `D1HOneLegStandCfg`

```python
class D1HOneLegStandCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent = 2 + 1 + 4 + 1 + 1 + 8 + 8 + 8  # 与flat一致
        n_proprio = 33
        history_len = 10
        num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent
        num_actions = 8  # 仍然控制8个关节

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # 略低于双腿站立
        rot = [0, 0.0, 0.0, 1]
        default_joint_angles = {
            'FL_hip_joint': 0,
            'FR_hip_joint': 0,
            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'FL_foot_joint': 0,
            'FR_foot_joint': 0,
        }
        # 抬起腿目标角度
        lift_leg_target = {
            'hip': 0.0,
            'thigh': 1.2,
            'calf': -2.5,
        }
        desired_feet_distance = 0.4

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip': 40., 'thigh': 40., 'calf': 40., 'foot': 10.}
        damping = {'hip': 1.0, 'thigh': 1.0, 'calf': 1.0, 'foot': 0.5}
        action_scale = 0.3  # 比行走更小（原0.5），站立需要精细控制
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands(LeggedRobotCfg.control):
        curriculum = False  # 站立技能不需要速度课程
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        class ranges:
            lin_vel_x = [-0.3, 0.3]  # 很小的速度范围
            lin_vel_y = [0.0, 0.0]    # 不允许横向移动
            ang_vel_yaw = [-0.3, 0.3] # 很小的转向范围
            heading = [-3.14, 3.14]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = True

    class rewards:
        # ... 如上表所示的所有奖励权重
        base_height_target = 0.45
        tracking_sigma = 0.25
        distance_sigma = 0.1
        # ...

    class costs(LeggedRobotCfg.costs):
        num_costs = 4
        class scales:
            pos_limit = 0.3
            torque_limit = 0.3
            dof_vel_limits = 0.3
            default_joint = 0.3
```

#### 4.1.3 Play 配置类 `D1HOneLegStandCfg_Play`

继承 `D1HOneLegStandCfg`，关闭噪声、域随机化，减少环境数。

#### 4.1.4 PPO 配置类 `D1HOneLegStandCfgPPO`

与 `D1HFlatCfgPPO` 基本一致，修改：
- `experiment_name = 'd1h_one_leg_stand'`
- `num_costs = 4`（增加了 default_joint cost）

---

### 4.2 修改 `configs/d1h/__init__.py`

新增两行注册：

```python
from .d1h_one_leg_stand_config import *
task_registry.register("d1h_one_leg_stand", D1HOneLegStand, D1HOneLegStandCfg(), D1HOneLegStandCfgPPO())
task_registry.register("d1h_one_leg_stand_play", D1HOneLegStand, D1HOneLegStandCfg_Play(), D1HOneLegStandCfgPPO())
```

---

## 5. 训练命令

```bash
python scripts/train.py --task=d1h_one_leg_stand --sim_device=cuda:0 --rl_device=cuda:0 --headless --max_iterations=50000
```

评估：
```bash
python scripts/simple_play.py --task=d1h_one_leg_stand_play --rl_device=cuda:0 --sim_device=cuda:0
```

---

## 6. 训练策略建议

### 6.1 课程学习（Curriculum）

建议分两阶段训练：

**阶段1（0-20000 iter）**：宽松条件
- 抬起腿触地惩罚较轻（`lift_foot_no_contact = 3.0`）
- 初始姿态随机范围小（roll/pitch ±0.2）
- 终止条件宽松（projected_gravity_z < 0.3 即终止）

**阶段2（20000-50000 iter）**：严格条件
- 抬起腿触地惩罚增强（`lift_foot_no_contact = 8.0`）
- 初始姿态随机范围扩大（roll/pitch ±0.5）
- 终止条件收紧（projected_gravity_z < 0.5 即终止）

可通过在 `_reset_root_states` 中根据 `self.tot_iter` 动态调整随机范围实现。

### 6.2 域随机化

训练时建议开启以下随机化（帮助 sim2real 迁移）：
- 摩擦系数随机化
- 质量随机化
- 推力扰动（`push_robots = True`，但力度减小）
- KP/KD 随机化

### 6.3 关键超参数

| 参数 | 建议值 | 原因 |
|------|--------|------|
| `action_scale` | 0.3 | 站立需要精细控制 |
| `max_episode_length` | 10s (4000步) | 比行走短，站立成功应快速稳定 |
| `entropy_coef` | 0.02 | 略高于行走，鼓励探索 |
| `learning_rate` | 1e-3 | 与行走一致 |

---

## 7. 实现步骤清单

- [ ] **Step 1**：创建 `configs/d1h/d1h_one_leg_stand_config.py`
  - [ ] 实现 `D1HOneLegStand` 环境类
  - [ ] 实现 `_init_buffers`（含站立/抬起腿索引）
  - [ ] 实现 `_reset_root_states`（随机选择站立腿，缩小初始随机范围）
  - [ ] 实现 `_reset_dofs`（抬起腿设置收起姿态）
  - [ ] 实现 `check_termination`（新增抬起腿触地、倾斜过大终止）
  - [ ] 实现新增奖励函数（stand_foot_contact, lift_foot_no_contact, lift_leg_target, balance_center）
  - [ ] 实现配置类 `D1HOneLegStandCfg`
  - [ ] 实现配置类 `D1HOneLegStandCfg_Play`
  - [ ] 实现配置类 `D1HOneLegStandCfgPPO`
- [ ] **Step 2**：修改 `configs/d1h/__init__.py`，注册新任务
- [ ] **Step 3**：运行 `python scripts/test_env.py --task=d1h_one_leg_stand` 验证环境创建无报错
- [ ] **Step 4**：启动训练并监控奖励曲线
- [ ] **Step 5**：根据训练效果调参（奖励权重、课程策略等）

---

## 8. 风险与注意事项

1. **轮子平衡 vs 关节刚度**：单腿站立时轮子是唯一平衡手段，`foot` 关节的 KP/KD 需要足够大才能提供平衡力矩，但过大会导致震荡。建议从 KP=10 开始，若平衡困难可尝试 KP=20。
2. **抬起腿的轮子**：抬起腿的轮子在空中自由旋转无意义，建议在 `step()` 中将抬起腿的 `foot_joint` 位置也置零（与站立腿一致），让PD控制器将其锁定。
3. **对称性**：通过随机选择站立腿（左/右），策略可以学会左右腿都能站立。
4. **观测空间一致性**：保持与行走技能相同的观测维度，便于未来技能组合/切换。
