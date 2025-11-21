import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import json
from typing import List, Dict, Tuple, Set, Optional, Any
import networkx as nx
from collections import defaultdict
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize_scalar

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 自定义颜色映射
layer_colors = ['#FF6B6B', '#FFD700', '#45B7D1', '#FFA07A', '#98D8C8']
cmap = LinearSegmentedColormap.from_list('layer_cmap', layer_colors, N=len(layer_colors))

# 默认的 spacing guard 比例（0 表示边缘框与模块紧贴）
DEFAULT_SPACING_GUARD_RATIO = 0.0



class PhysicalModule:
    """物理模块类，表示布线中的基本功能模块"""

    def __init__(self, module_id: str, x: float, y: float, width: float, height: float,
                 module_type: str, layer: int, pins: List[Tuple[float, float]], **kwargs):
        self.id = module_id
        self.x = x  # 左下角x坐标
        self.y = y  # 左下角y坐标
        self.width = width
        self.height = height
        self.type = module_type  # clk, analog, digital, memory, io, obstacle
        self.layer = layer  # 所在层
        self.kwargs = kwargs

        # 关键修复：添加pins参数的严格校验
        self.pins = self._validate_pins(pins)
        self.adjacent_modules = set()  # 相邻模块ID
        
        # ========== 新增：DRC相关属性 ==========
        # 模块敏感度类型（用于DRC间距规则）
        # 'digital': 数字宏（CPU核心、GPU、NPU、SRAM、缓存控制器、DSP、视频编解码器等）
        # 'sensitive': 敏感宏（PLL、电源管理、时钟、安全引擎、USB、PCIe、模拟ADC等）
        self.sensitivity = kwargs.get('sensitivity', self._infer_sensitivity(module_type))
        
        # 边缘框属性（用于初始布局阶段）
        self.spacing_guard_active = True  # 边缘框是否激活（压缩后废弃）
        self.spacing_guard_ratio = kwargs.get('spacing_guard_ratio', DEFAULT_SPACING_GUARD_RATIO)
        self.spacing_guard_width = self._calculate_spacing_guard()  # 边缘框宽度

        # 力学属性
        self.velocity_x = kwargs.get('velocity_x', 0.0)  # x方向速度
        self.velocity_y = kwargs.get('velocity_y', 0.0)  # y方向速度
        self.force_x = kwargs.get('force_x', 0.0)  # x方向受力
        self.force_y = kwargs.get('force_y', 0.0)  # y方向受力
        self.mass = kwargs.get('mass', width * height)  # 质量，默认为面积
        self.damping = kwargs.get('damping', 0.3)  # 阻尼系数

        # 根据模块类型设置默认的电荷和弹簧常数
        type_defaults = {
            'clk': {'charge': 2.0, 'spring_constant': 0.8, 'is_fixed': False},
            'analog': {'charge': 1.5, 'spring_constant': 0.6, 'is_fixed': False},
            'memory': {'charge': 1.2, 'spring_constant': 0.7, 'is_fixed': False},
            'io': {'charge': 0.8, 'spring_constant': 0.5, 'is_fixed': True},
            'digital': {'charge': 1.0, 'spring_constant': 0.5, 'is_fixed': False},
            'obstacle': {'charge': 3.0, 'spring_constant': 0.0, 'is_fixed': True}
        }

        # 获取该类型的默认值
        defaults = type_defaults.get(module_type, {'charge': 1.0, 'spring_constant': 0.5, 'is_fixed': False})

        # 设置电荷和弹簧常数，允许通过kwargs覆盖
        self.charge = kwargs.get('charge', defaults['charge'])
        self.spring_constant = kwargs.get('spring_constant', defaults['spring_constant'])
        self.is_fixed = kwargs.get('is_fixed', defaults['is_fixed'])

    def _validate_pins(self, pins: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """验证pins格式，确保每个元素都是包含两个数值的可迭代对象"""
        validated = []
        for idx, pin in enumerate(pins):
            # 检查是否为可迭代对象且长度为2
            if not isinstance(pin, (list, tuple)):
                raise TypeError(f"引脚 #{idx} 类型错误：应为元组/列表，实际为 {type(pin).__name__}")
            if len(pin) != 2:
                raise ValueError(f"引脚 #{idx} 长度错误：应包含2个坐标值，实际为 {len(pin)} 个")
            # 检查坐标是否为数值类型
            if not all(isinstance(coord, (int, float)) for coord in pin):
                raise TypeError(f"引脚 #{idx} 坐标类型错误：应为数值，实际为 {[type(c).__name__ for c in pin]}")
            validated.append((float(pin[0]), float(pin[1])))
        return validated
    
    def _infer_sensitivity(self, module_type: str) -> str:
        """
        根据模块类型推断敏感度
        
        返回:
            'digital': 数字宏
            'sensitive': 敏感宏
        """
        # 敏感模块类型列表（PLL、电源管理、时钟、模拟等）
        sensitive_types = {'clk', 'analog', 'io'}
        
        if module_type in sensitive_types:
            return 'sensitive'
        else:
            # 默认为数字宏（digital, memory等）
            return 'digital'
    
    def _calculate_spacing_guard(self) -> float:
        """
        计算边缘框宽度（Spacing Guard）
        
        默认情况下取 0，使边缘框与模块边界紧贴。
        如需扩展边缘框，可在构造函数中传入 spacing_guard_ratio。
        """
        min_edge = min(self.width, self.height)
        return min_edge * max(0.0, float(self.spacing_guard_ratio))
    
    def get_bounds_with_guard(self) -> Tuple[float, float, float, float]:
        """
        获取包含边缘框的模块边界
        
        返回:
            (min_x, max_x, min_y, max_y) - 包含边缘框的边界
        """
        if self.spacing_guard_active:
            guard = self.spacing_guard_width
            return (
                self.x - guard,
                self.x + self.width + guard,
                self.y - guard,
                self.y + self.height + guard
            )
        else:
            # 边缘框未激活时，返回实际边界
            return self.get_bounds()

    def get_pin_absolute(self, pin_idx):
        """获取引脚的绝对坐标"""
        # 检查pins是否存在且不为空
        if not self.pins or len(self.pins) == 0:
            # 如果没有引脚，返回模块中心
            return (self.x + self.width / 2, self.y + self.height / 2)

        # 如果pin_idx是字符串，转换为整数索引
        if isinstance(pin_idx, str):
            # 根据字符串映射到引脚索引
            pin_mapping = {"in": 0, "out": 1, "default": 0}
            pin_idx = pin_mapping.get(pin_idx, 0)

        # 确保pin_idx是整数且在有效范围内
        try:
            # 修复：确保访问的是经过校验的元组
            px, py = self.pins[pin_idx]
            return (self.x + px, self.y + py)
        except IndexError:
            raise IndexError(f"引脚索引 {pin_idx} 超出范围，模块 {self.id} 仅有 {len(self.pins)} 个引脚")

    def get_center(self):
        """获取模块中心坐标"""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def get_bounds(self):
        """获取模块边界 (min_x, max_x, min_y, max_y)"""
        return (self.x, self.x + self.width, self.y, self.y + self.height)

    def contains_point(self, x: float, y: float) -> bool:
        """判断点是否在模块内部（用于碰撞检测）"""
        x_min, x_max, y_min, y_max = self.get_bounds()
        return x_min <= x <= x_max and y_min <= y <= y_max

    def apply_force(self, fx: float, fy: float):
        """施加力到模块上"""
        if not self.is_fixed:
            self.force_x += fx
            self.force_y += fy

    def reset_forces(self):
        """重置受力"""
        self.force_x = 0.0
        self.force_y = 0.0

    def update_velocity(self, dt: float):
        """根据受力更新速度（物理模拟）"""
        if not self.is_fixed and self.mass > 0:
            # F = ma, 所以 a = F/m
            ax = self.force_x / self.mass
            ay = self.force_y / self.mass

            # 更新速度：v = v0 + a*dt
            self.velocity_x += ax * dt
            self.velocity_y += ay * dt

            # 应用阻尼
            self.velocity_x *= (1 - self.damping)
            self.velocity_y *= (1 - self.damping)

    def update_position(self, dt: float):
        """根据速度更新位置"""
        if not self.is_fixed:
            # x = x0 + v*dt
            self.x += self.velocity_x * dt
            self.y += self.velocity_y * dt

    def get_kinetic_energy(self) -> float:
        """计算动能"""
        return 0.5 * self.mass * (self.velocity_x ** 2 + self.velocity_y ** 2)

    def get_mechanical_info(self) -> dict:
        """获取力学属性信息"""
        return {
            'id': self.id,
            'type': self.type,
            'position': (self.x, self.y),
            'velocity': (self.velocity_x, self.velocity_y),
            'force': (self.force_x, self.force_y),
            'mass': self.mass,
            'charge': self.charge,
            'spring_constant': self.spring_constant,
            'damping': self.damping,
            'is_fixed': self.is_fixed,
            'kinetic_energy': self.get_kinetic_energy()
        }
    
    def copy(self, new_x: float = None, new_y: float = None) -> 'PhysicalModule':
        """
        创建模块的副本
        
        参数:
            new_x: 新的x坐标（可选，默认使用当前x）
            new_y: 新的y坐标（可选，默认使用当前y）
        
        返回:
            模块副本
        """
        new_module = PhysicalModule(
            module_id=self.id,
            x=new_x if new_x is not None else self.x,
            y=new_y if new_y is not None else self.y,
            width=self.width,
            height=self.height,
            module_type=self.type,
            layer=self.layer,
            pins=self.pins.copy(),
            sensitivity=self.sensitivity,  # 确保复制敏感度信息
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
            force_x=self.force_x,
            force_y=self.force_y,
            mass=self.mass,
            damping=self.damping,
            charge=self.charge,
            spring_constant=self.spring_constant,
            is_fixed=self.is_fixed
        )
        # 复制边框激活状态和宽度
        new_module.spacing_guard_active = self.spacing_guard_active
        new_module.spacing_guard_ratio = self.spacing_guard_ratio
        new_module.spacing_guard_width = self.spacing_guard_width
        return new_module

    # ========== 阶段2：间距规则引擎 ==========
    
    @staticmethod
    def calculate_required_spacing(moduleA: 'PhysicalModule', moduleB: 'PhysicalModule') -> float:
        """
        根据8%/24%规则计算DRC所需最小间距。

        规则：
        - 数字宏 ↔ 数字宏：max(0.08 × min_edge(A), 0.08 × min_edge(B))
        - 其他组合（含敏感宏、障碍物等）：max(0.24 × min_edge(A), 0.24 × min_edge(B))

        参数:
            moduleA: 第一个模块
            moduleB: 第二个模块

        返回:
            所需的最小间距（单位：μm）
        """
        # 判定是否数字-数字组合（其余均视为“其他组合”）
        is_digital_pair = (getattr(moduleA, 'sensitivity', 'digital') == 'digital' and
                           getattr(moduleB, 'sensitivity', 'digital') == 'digital')

        ratio = 0.08 if is_digital_pair else 0.24
        min_edge_a = min(moduleA.width, moduleA.height)
        min_edge_b = min(moduleB.width, moduleB.height)
        req_a = min_edge_a * ratio
        req_b = min_edge_b * ratio
        return max(req_a, req_b)
    
    def calculate_edge_distance(self, other: 'PhysicalModule') -> float:
        """
        计算两个模块边缘到边缘的最短距离（边-边距离）
        
        如果模块重叠，返回负值表示重叠深度
        
        参数:
            other: 另一个模块
        
        返回:
            边缘到边缘的距离（正数=间隙，负数=重叠）
        """
        # 获取两个模块的边界
        x1_min, x1_max, y1_min, y1_max = self.get_bounds()
        x2_min, x2_max, y2_min, y2_max = other.get_bounds()
        
        # 计算x方向和y方向的距离
        dx = 0.0
        dy = 0.0
        
        # x方向距离
        if x1_max <= x2_min:
            dx = x2_min - x1_max  # self在左，other在右
        elif x2_max <= x1_min:
            dx = x1_min - x2_max  # other在左，self在右
        else:
            # x方向重叠
            dx = -min(x1_max - x2_min, x2_max - x1_min)
        
        # y方向距离
        if y1_max <= y2_min:
            dy = y2_min - y1_max  # self在下，other在上
        elif y2_max <= y1_min:
            dy = y1_min - y2_max  # other在下，self在上
        else:
            # y方向重叠
            dy = -min(y1_max - y2_min, y2_max - y1_min)
        
        # 如果两个方向都没有重叠，返回欧几里得距离
        if dx >= 0 and dy >= 0:
            return math.sqrt(dx**2 + dy**2)
        # 如果只有一个方向重叠，返回另一个方向的距离
        elif dx < 0 and dy >= 0:
            return dy
        elif dx >= 0 and dy < 0:
            return dx
        # 如果两个方向都重叠，返回较小的重叠深度（负值）
        else:
            return max(dx, dy)
    
    def check_drc_violation(self, other: 'PhysicalModule') -> Tuple[bool, float, float]:
        """
        检查与另一个模块是否存在DRC违规
        
        参数:
            other: 另一个模块
        
        返回:
            (是否违规, 所需间距, 实际间距)
        """
        required_spacing = self.calculate_required_spacing(self, other)
        actual_distance = self.calculate_edge_distance(other)
        
        # 如果实际间距小于所需间距，则违规
        is_violation = actual_distance < required_spacing
        
        return is_violation, required_spacing, actual_distance


class LayoutOptimizer:
    """基于最小生成树的多智能体布局优化器，以HPWL为唯一衡量指标"""

    def __init__(self, modules: List[PhysicalModule], connections: List[Tuple], num_agents: int = 5,
                 min_spacing: float = 15.0, enforce_zero_module_gaps: bool = True):
        self.modules = {m.id: m for m in modules}
        self.connections = [
            conn for conn in connections
            if len(conn) >= 4 and  # 确保有完整的端点信息
               conn[0] in self.modules and  # 源模块存在
               conn[2] in self.modules and  # 目的模块存在
               conn[0] != conn[2]  # 排除自连接
        ]
        self.original_positions = {m.id: (m.x, m.y) for m in modules}
        self.min_spacing = min_spacing  # 最小间距要求（单位：um）

        # 多智能体布局参数
        self.num_agents = num_agents
        self.layout_agents = self._initialize_layout_agents()

        # 力学模拟参数
        self.damping = 0.5  # 阻尼系数
        self.time_step = 0.1  # 时间步长
        self.max_velocity = 50.0  # 最大速度限制
        self.coulomb_constant = 1000.0  # 库仑常数
        self.iteration = 0  # 迭代计数器
        self.enforce_zero_module_gaps = enforce_zero_module_gaps
        
        # 布局尺寸参数
        self.layout_width = 200.0  # 布局宽度
        self.layout_height = 100.0  # 布局高度

    def _initialize_layout_agents(self) -> List[Dict]:
        """初始化多个布局智能体，每个使用不同的参数组合和随机种子，偏向紧凑布局"""
        import random  # 确保随机可用

        # 基础布局策略（调整为更紧凑，减小base_k以使模块尽量近）
        base_strategies = [
            # 策略1: 紧凑标准力导向
            {
                "name": "CompactStandard",
                "base_k": 1.5,  # 减小k，更紧凑
                "base_iterations": 100,
                "weight_factor": 1.2,  # 加强边吸引
                "random_k_range": 0.2,
                "random_iterations_var": 10,
                "attract_factor": 0.8  # 拉近强度
            },
            # 策略2: 超紧凑布局
            {
                "name": "UltraCompact",
                "base_k": 1.0,  # 进一步减小k
                "base_iterations": 80,
                "weight_factor": 1.5,  # 强吸引
                "random_k_range": 0.3,
                "random_iterations_var": 15,
                "attract_factor": 1.0
            },
            # 策略3: 平衡扩展（稍松但仍紧凑）
            {
                "name": "BalancedCompact",
                "base_k": 2.0,  # 适中
                "base_iterations": 120,
                "weight_factor": 1.0,
                "random_k_range": 0.4,
                "random_iterations_var": 20,
                "attract_factor": 0.6
            },
            # 策略4: 快速紧凑
            {
                "name": "FastCompact",
                "base_k": 1.2,
                "base_iterations": 50,
                "weight_factor": 1.3,
                "random_k_range": 0.1,
                "random_iterations_var": 5,
                "attract_factor": 0.9
            },
            # 策略5: 高精度紧凑
            {
                "name": "PreciseCompact",
                "base_k": 1.3,
                "base_iterations": 150,
                "weight_factor": 1.1,
                "random_k_range": 0.15,
                "random_iterations_var": 15,
                "attract_factor": 0.7
            },
            # 策略6: 随机紧凑探索
            {
                "name": "RandomCompact",
                "base_k": 1.2,
                "base_iterations": 100,
                "weight_factor": 1.4,
                "random_k_range": 0.5,
                "random_iterations_var": 30,
                "attract_factor": 1.2  # 强拉近
            },
            # 策略7: HPWL优化策略
            {
                "name": "HPWLOptimizer",
                "base_k": 0.3,  # 极小的理想距离
                "base_iterations": 300,
                "weight_factor": 0.6,
                "random_k_range": 0.1,
                "random_iterations_var": 20,
                "attract_factor": 0.8  # 专注于HPWL优化
            },
            # 策略8: 连接权重策略
            {
                "name": "ConnectionWeight",
                "base_k": 0.4,  # 基于连接权重调整距离
                "base_iterations": 250,
                "weight_factor": 0.7,
                "random_k_range": 0.15,
                "random_iterations_var": 25,
                "attract_factor": 0.9  # 平衡HPWL和密度
            }
        ]

        # 根据代理数量选择并变异策略
        selected_strategies = base_strategies[:self.num_agents] if self.num_agents <= len(
            base_strategies) else base_strategies

        agents = []
        random.seed(42)  # 固定种子以确保可重复性，但每个代理有子种子
        for i in range(self.num_agents):
            strategy = selected_strategies[min(i, len(selected_strategies) - 1)].copy()

            # 应用随机变异（偏向更小k）
            strategy["k"] = strategy["base_k"] + random.uniform(-strategy["random_k_range"],
                                                                strategy["random_k_range"] * 0.5)  # 偏向减小k
            strategy["iterations"] = max(20, strategy["base_iterations"] + random.randint(
                -strategy["random_iterations_var"], strategy["random_iterations_var"]))

            # 唯一随机种子（基于代理ID）
            agent_seed = 42 + i * 100  # 确定性但不同
            random.seed(agent_seed)
            strategy["seed"] = agent_seed

            agent = {
                "id": i,
                "strategy": strategy,
                "hpwl_history": [],  # 记录每个代理的HPWL历史
                "best_layout": None,  # 最佳布局
                "best_hpwl": float('inf')  # 最佳HPWL
            }
            agents.append(agent)

        print(f"初始化 {self.num_agents} 个布局智能体（紧凑偏好 + 随机扰动）:")
        for agent in agents:
            print(
                f"  智能体{agent['id']}: {agent['strategy']['name']} (k={agent['strategy']['k']:.2f}, 迭代={agent['strategy']['iterations']}, seed={agent['strategy']['seed']})")

        return agents

    @staticmethod
    def _get_effective_bounds(module: PhysicalModule) -> Tuple[float, float, float, float]:
        """
        返回用于间距/重叠计算的真实边界：
        - spacing guard 激活时使用包含 guard 的边界
        - 否则使用模块本体边界
        """
        if getattr(module, 'spacing_guard_active', False):
            return module.get_bounds_with_guard()
        return module.get_bounds()

    @staticmethod
    def _interval_overlap_length(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
        """计算两个一维区间的重叠长度（可为负表示间隙）"""
        return min(a_max, b_max) - max(a_min, b_min)

    @staticmethod
    def _module_movable(module: PhysicalModule) -> bool:
        """判断模块在当前阶段是否允许移动"""
        if getattr(module, 'is_fixed', False):
            return False
        return module.type != 'obstacle'

    def _try_shift_module_along_axis(
        self,
        module: PhysicalModule,
        delta: float,
        axis: str,
        all_modules: List[PhysicalModule],
        max_attempts: int = 8
    ) -> bool:
        """
        尝试将模块沿某轴移动 delta（正=正向，负=反向）
        成功时直接更新 module 的位置
        """
        if abs(delta) < 1e-12 or not self._module_movable(module):
            return False

        layout_w = getattr(self, 'layout_width', None)
        layout_h = getattr(self, 'layout_height', None)

        shift = delta
        for _ in range(max_attempts):
            new_x = module.x + (shift if axis == 'x' else 0.0)
            new_y = module.y + (shift if axis == 'y' else 0.0)

            if layout_w is not None and layout_w > 0:
                new_x = max(0.0, min(layout_w - module.width, new_x))
            if layout_h is not None and layout_h > 0:
                new_y = max(0.0, min(layout_h - module.height, new_y))

            temp_module = module.copy(new_x=new_x, new_y=new_y)
            has_overlap, _ = self.check_overlap_with_all_modules(
                temp_module,
                all_modules,
                exclude_ids={module.id}
            )
            if not has_overlap:
                module.x = temp_module.x
                module.y = temp_module.y
                return True

            shift *= 0.5

        return False

    def _enforce_zero_guard_gaps_complete(
        self,
        modules: List[PhysicalModule],
        tolerance: float = 1e-9,
        max_rounds: int = 6
    ) -> List[PhysicalModule]:
        """
        完整的零缝隙压实流程：组合使用多种压实策略，确保守护边框之间完全零缝隙
        
        参数:
            modules: 当前模块列表
            tolerance: 允许的最大缝隙（默认1e-9，接近零）
            max_rounds: 最大迭代轮数
        
        返回:
            压实后的模块列表
        """
        active_modules = [m for m in modules if getattr(m, 'spacing_guard_active', False)]
        if len(active_modules) < 2:
            return modules
        
        print(f"  执行完整边缘框零缝隙压实流程（容差={tolerance:.1e}，最大轮数={max_rounds}）...")
        print(f"  目标：使所有边缘框之间的缝隙为0（不是模块本身之间的缝隙为0）")
        
        # 记录初始最大边缘框缝隙
        initial_max_gap = 0.0
        for i in range(len(active_modules)):
            for j in range(i + 1, len(active_modules)):
                gap = self.calculate_module_gap(active_modules[i], active_modules[j])
                if gap > initial_max_gap:
                    initial_max_gap = gap
        
        if initial_max_gap <= tolerance:
            print(f"  [跳过] 已满足边缘框零缝隙要求（最大边缘框缝隙={initial_max_gap:.1e}um <= {tolerance:.1e}um）")
            return modules
        
        # 策略1：消除所有边缘框缝隙（主要策略）
        print("    策略1: 消除所有边缘框缝隙...")
        modules = self.eliminate_spacing_guard_gaps(modules, tolerance=tolerance, max_iterations=1500)
        
        # 策略2：处理边缘框斜向缝隙
        print("    策略2: 处理边缘框斜向缝隙...")
        modules = self.snap_guard_pairs(modules, tolerance=tolerance, max_rounds=max_rounds * 2)
        
        # 策略3：按轴全面压实边缘框
        print("    策略3: 按轴全面压实边缘框...")
        modules = self.enforce_zero_guard_gaps(modules, tolerance=tolerance, max_rounds=max_rounds * 2)
        
        # 策略4：迭代压实边缘框（多轮迭代，直到收敛）
        print("    策略4: 迭代压实边缘框（直到收敛）...")
        for round_idx in range(max_rounds * 2):
            # 计算当前最大缝隙
            current_max_gap = 0.0
            gap_pairs = []
            
            for i in range(len(modules)):
                mod_i = modules[i]
                if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                    continue
                
                for j in range(i + 1, len(modules)):
                    mod_j = modules[j]
                    if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                        continue
                    
                    gap = self.calculate_module_gap(mod_i, mod_j)
                    if gap > tolerance:
                        gap_pairs.append((gap, mod_i, mod_j))
                        current_max_gap = max(current_max_gap, gap)
            
            if current_max_gap <= tolerance:
                print(f"    策略4: 第{round_idx+1}轮后已收敛（最大边缘框缝隙={current_max_gap:.1e}um）")
                break
            
            # 按缝隙大小排序，优先处理大缝隙
            gap_pairs.sort(key=lambda x: x[0], reverse=True)
            
            moves_this_round = 0
            def movable(module: PhysicalModule) -> bool:
                return module.type != 'obstacle' and not getattr(module, 'is_fixed', False)
            
            for gap, mod_a, mod_b in gap_pairs[:min(100, len(gap_pairs))]:  # 每轮最多处理100对，增强压缩能力
                if gap <= tolerance:
                    continue
                
                # 选择可移动模块
                mover, reference = None, None
                if movable(mod_a):
                    mover, reference = mod_a, mod_b
                elif movable(mod_b):
                    mover, reference = mod_b, mod_a
                else:
                    continue
                
                # 尝试拉近
                moved = self.move_module_closer(
                    mover,
                    reference,
                    target_spacing=tolerance,
                    all_modules=modules
                )
                
                if moved and (abs(moved.x - mover.x) > 1e-12 or abs(moved.y - mover.y) > 1e-12):
                    mover.x = moved.x
                    mover.y = moved.y
                    moves_this_round += 1
            
            if moves_this_round == 0:
                print(f"    策略4: 第{round_idx+1}轮无改进，提前结束")
                break
            
            print(f"    策略4: 第{round_idx+1}轮压实了{moves_this_round}对边缘框，最大剩余边缘框缝隙={current_max_gap:.6f}um")
            
            # 如果还有较大缝隙，再次调用策略1-3进行强化压缩
            if current_max_gap > tolerance * 10 and round_idx % 3 == 2:
                print(f"    策略4: 第{round_idx+1}轮后仍有较大缝隙，执行强化压缩...")
                modules = self.eliminate_spacing_guard_gaps(modules, tolerance=tolerance, max_iterations=500)
                modules = self.snap_guard_pairs(modules, tolerance=tolerance, max_rounds=3)
                modules = self.enforce_zero_guard_gaps(modules, tolerance=tolerance, max_rounds=3)
        
        # 最终强化压缩：再次执行所有策略确保完全收敛
        print("    策略5: 最终强化压缩（确保完全收敛）...")
        modules = self.eliminate_spacing_guard_gaps(modules, tolerance=tolerance, max_iterations=800)
        modules = self.snap_guard_pairs(modules, tolerance=tolerance, max_rounds=4)
        modules = self.enforce_zero_guard_gaps(modules, tolerance=tolerance, max_rounds=4)
        
        # 最终验证（验证边缘框间缝隙）
        final_max_gap = 0.0
        final_avg_gap = 0.0
        gap_count = 0
        
        for i in range(len(modules)):
            mod_i = modules[i]
            if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                continue
            
            for j in range(i + 1, len(modules)):
                mod_j = modules[j]
                if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                    continue
                
                gap = self.calculate_module_gap(mod_i, mod_j)
                if gap > 0:
                    final_max_gap = max(final_max_gap, gap)
                    final_avg_gap += gap
                    gap_count += 1
        
        if gap_count > 0:
            final_avg_gap /= gap_count
        
        improvement = initial_max_gap - final_max_gap
        print(f"  [完成] 边缘框零缝隙压实完成：")
        print(f"    初始最大边缘框缝隙: {initial_max_gap:.6f}um")
        print(f"    最终最大边缘框缝隙: {final_max_gap:.6f}um")
        print(f"    最终平均边缘框缝隙: {final_avg_gap:.6f}um")
        print(f"    改进: {improvement:.6f}um ({improvement/initial_max_gap*100:.2f}%)")
        print(f"  [重要] 注意：这是边缘框间的缝隙，不是模块本身之间的缝隙")
        
        if final_max_gap <= tolerance:
            print(f"  [成功] 所有边缘框缝隙已压缩至 <= {tolerance:.1e}um（边缘框保留，边缘框间缝隙为0）")
        else:
            print(f"  [警告] 仍有最大 {final_max_gap:.6f}um 的边缘框缝隙（可能受固定/障碍模块限制）")
        
        return modules

    def enforce_zero_guard_gaps(
        self,
        modules: List[PhysicalModule],
        tolerance: float = 1e-9,
        max_rounds: int = 4
    ) -> List[PhysicalModule]:
        """
        全面压实 spacing guard，使任意守护边框之间不留缝隙
        （仅在 spacing_guard_active=True 阶段使用）
        """
        active_modules = [m for m in modules if getattr(m, 'spacing_guard_active', False)]
        if len(active_modules) < 2:
            return modules

        print("执行守护边框零缝隙压实...")

        axes = [
            ('x', 0, 1, 2, 3),  # (axis, start_idx, end_idx, perp_start, perp_end)
            ('y', 2, 3, 0, 1),
        ]

        for axis_name, start_idx, end_idx, perp_start_idx, perp_end_idx in axes:
            for round_idx in range(1, max_rounds + 1):
                moved_this_round = 0
                need_resort = True

                while need_resort:
                    need_resort = False
                    sorted_active = sorted(
                        active_modules,
                        key=lambda m: self._get_effective_bounds(m)[start_idx]
                    )

                    for idx in range(1, len(sorted_active)):
                        prev = sorted_active[idx - 1]
                        curr = sorted_active[idx]
                        prev_bounds = self._get_effective_bounds(prev)
                        curr_bounds = self._get_effective_bounds(curr)

                        # 只有在垂直方向存在重叠时才需要压实
                        perp_overlap = self._interval_overlap_length(
                            prev_bounds[perp_start_idx],
                            prev_bounds[perp_end_idx],
                            curr_bounds[perp_start_idx],
                            curr_bounds[perp_end_idx]
                        )
                        if perp_overlap <= tolerance:
                            continue

                        gap = curr_bounds[start_idx] - prev_bounds[end_idx]
                        if gap <= tolerance:
                            continue

                        moved = False
                        # 优先将后一个模块向前滑动
                        if self._try_shift_module_along_axis(curr, -gap, axis_name, modules):
                            moved = True
                        # 如果无法移动，尝试推动前一个模块
                        elif self._try_shift_module_along_axis(prev, gap, axis_name, modules):
                            moved = True

                        if moved:
                            moved_this_round += 1
                            need_resort = True
                            break  # 重新排序后再处理

                print(
                    f"  [{axis_name}-axis] round {round_idx}: 压实 {moved_this_round} 次"
                )
                if moved_this_round == 0:
                    break

        # 统计压实后的最大残余缝隙
        max_gap = 0.0
        for i in range(len(active_modules)):
            for j in range(i + 1, len(active_modules)):
                gap = self.calculate_module_gap(active_modules[i], active_modules[j])
                if gap > max_gap:
                    max_gap = gap

        if max_gap <= tolerance:
            print(f"  [成功] 守护边框缝隙全部压缩至 <= {tolerance:.1e}um")
        else:
            print(f"  [提示] 仍存在最大 {max_gap:.3e}um 的守护缝隙（可能受固定/障碍模块限制）")

        return modules

    def _calculate_layout_hpwl(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """计算布局的HPWL（标准HPWL算法：网表最小包围盒）"""
        from collections import defaultdict
        
        total_hpwl = 0.0
        
        # 构建网表：将连接关系转换为网表
        nets = defaultdict(set)  # 使用set避免重复引脚
        
        for connection in self.connections:
            if len(connection) >= 2:
                # 处理两种连接格式
                if len(connection) == 4:
                    # 格式: (src_id, src_pin, dst_id, dst_pin)
                    module1_id, module2_id = connection[0], connection[2]
                elif len(connection) == 2:
                    # 格式: (src_id, dst_id)
                    module1_id, module2_id = connection[0], connection[1]
                else:
                    continue
                
                if module1_id in self.modules and module2_id in self.modules:
                    # 为每个连接创建唯一的网表ID
                    net_id = f"net_{module1_id}_{module2_id}"
                    nets[net_id].add(module1_id)
                    nets[net_id].add(module2_id)
        
        # 计算每个网表的HPWL
        for net_id, module_ids in nets.items():
            if len(module_ids) >= 2:
                # 收集网表中所有模块的引脚坐标
                pin_coords = []
                
                for module_id in module_ids:
                    module = self.modules[module_id]
                    
                    # 使用传入的位置参数，如果没有则使用模块当前位置
                    if module_id in positions:
                        module_x, module_y = positions[module_id]
                    else:
                        module_x, module_y = module.x, module.y
                    
                    # 获取模块的实际引脚坐标（相对于模块左下角）
                    for pin_rel_x, pin_rel_y in module.pins:
                        # 转换为全局坐标
                        pin_global_x = module_x + pin_rel_x
                        pin_global_y = module_y + pin_rel_y
                        pin_coords.append((pin_global_x, pin_global_y))
                
                if len(pin_coords) >= 2:
                    # 收集所有引脚的x和y坐标
                    x_coords = [pin[0] for pin in pin_coords]
                    y_coords = [pin[1] for pin in pin_coords]
                    
                    # 计算最小包围盒
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    
                    # 标准HPWL公式：HPWL = (Xmax - Xmin) + (Ymax - Ymin)
                    hpwl = (xmax - xmin) + (ymax - ymin)
                    total_hpwl += hpwl
        
        return total_hpwl

    def _calculate_connection_weights(self) -> Dict[Tuple[str, str], float]:
        """计算连接权重，用于HPWL优化"""
        connection_weights = {}
        for src_id, _, dst_id, _ in self.connections:
            if src_id not in self.modules or dst_id not in self.modules:
                continue
            
            # 计算连接频率
            frequency = sum(1 for conn in self.connections 
                          if (conn[0] == src_id and conn[2] == dst_id) or 
                             (conn[0] == dst_id and conn[2] == src_id))
            
            # 计算模块重要性（基于连接数）
            src_connections = sum(1 for conn in self.connections if conn[0] == src_id or conn[2] == src_id)
            dst_connections = sum(1 for conn in self.connections if conn[0] == dst_id or conn[2] == dst_id)
            
            # 权重 = 连接频率 × 模块重要性
            weight = frequency * (src_connections + dst_connections)
            connection_weights[(src_id, dst_id)] = weight
            
        return connection_weights

    def _optimize_hpwl_with_weights(self, modules: List[PhysicalModule], 
                                   connection_weights: Dict[Tuple[str, str], float]) -> List[PhysicalModule]:
        """基于连接权重的HPWL优化"""
        print("执行基于连接权重的HPWL优化...")
        
        # 按权重排序连接
        sorted_connections = sorted(connection_weights.items(), key=lambda x: x[1], reverse=True)
        
        # 对高权重连接进行优化
        for (src_id, dst_id), weight in sorted_connections:
            if weight > 2:  # 只优化高权重连接
                src_module = next((m for m in modules if m.id == src_id), None)
                dst_module = next((m for m in modules if m.id == dst_id), None)
                
                if src_module and dst_module:
                    # 计算当前距离
                    current_distance = abs(dst_module.get_center()[0] - src_module.get_center()[0]) + \
                                     abs(dst_module.get_center()[1] - src_module.get_center()[1])
                    
                    # 如果距离过大，尝试拉近
                    if current_distance > 20.0:  # 距离阈值
                        self._pull_modules_closer(src_module, dst_module, modules)
        
        return modules

    def _pull_modules_closer(self, src_module: PhysicalModule, dst_module: PhysicalModule, 
                           modules: List[PhysicalModule]) -> None:
        """将两个模块拉近以减少HPWL"""
        # 计算目标位置（中点）
        src_center = src_module.get_center()
        dst_center = dst_module.get_center()
        
        target_x = (src_center[0] + dst_center[0]) / 2
        target_y = (src_center[1] + dst_center[1]) / 2
        
        # 计算移动距离
        move_factor = 0.3  # 移动因子
        
        # 移动源模块
        new_src_x = src_module.x + (target_x - src_center[0]) * move_factor
        new_src_y = src_module.y + (target_y - src_center[1]) * move_factor
        
        # 移动目标模块
        new_dst_x = dst_module.x + (target_x - dst_center[0]) * move_factor
        new_dst_y = dst_module.y + (target_y - dst_center[1]) * move_factor
        
        # 检查边界约束
        new_src_x = max(0, min(self.layout_width - src_module.width, new_src_x))
        new_src_y = max(0, min(self.layout_height - src_module.height, new_src_y))
        new_dst_x = max(0, min(self.layout_width - dst_module.width, new_dst_x))
        new_dst_y = max(0, min(self.layout_height - dst_module.height, new_dst_y))
        
        # 应用新位置
        src_module.x = new_src_x
        src_module.y = new_src_y
        dst_module.x = new_dst_x
        dst_module.y = new_dst_y

    def _hierarchical_hpwl_optimization(self, modules: List[PhysicalModule]) -> List[PhysicalModule]:
        """分层HPWL优化策略"""
        print("执行分层HPWL优化...")
        
        # 第一层：全局优化 - 基于连接权重
        connection_weights = self._calculate_connection_weights()
        modules = self._optimize_hpwl_with_weights(modules, connection_weights)
        
        # 第二层：局部优化 - 相邻模块聚类
        modules = self._cluster_adjacent_modules(modules)
        
        # 第三层：精细优化 - 微调位置
        modules = self._fine_tune_positions(modules)
        
        return modules
    
    def _optimize_hpwl_by_connection_graph(self, modules: List[PhysicalModule], max_iterations: int = 3) -> List[PhysicalModule]:
        """基于连接图的HPWL优化：拉近高权重连接的模块"""
        print(f"  基于连接图优化HPWL（{max_iterations}轮）...")
        
        module_map = {m.id: m for m in modules}
        connection_weights = self._calculate_connection_weights()
        
        # 按权重排序连接
        sorted_connections = sorted(connection_weights.items(), key=lambda x: x[1], reverse=True)
        
        initial_hpwl = self._calculate_current_hpwl(modules)
        
        for iteration in range(max_iterations):
            improved = False
            
            # 处理高权重连接（权重 > 1）
            for (src_id, dst_id), weight in sorted_connections:
                if weight <= 1:
                    continue
                    
                src_module = module_map.get(src_id)
                dst_module = module_map.get(dst_id)
                
                if not src_module or not dst_module:
                    continue
                
                # 计算当前HPWL贡献（简化：使用曼哈顿距离）
                src_center = src_module.get_center()
                dst_center = dst_module.get_center()
                current_distance = abs(dst_center[0] - src_center[0]) + abs(dst_center[1] - src_center[1])
                
                # 如果距离较大，尝试拉近
                if current_distance > 15.0:
                    # 计算拉近方向
                    dx = dst_center[0] - src_center[0]
                    dy = dst_center[1] - src_center[1]
                    distance = max(math.sqrt(dx**2 + dy**2), 1e-5)
                    
                    # 拉近因子（随迭代递减）
                    pull_factor = 0.4 * (1.0 - iteration / max_iterations * 0.5)
                    
                    # 计算新位置（向对方移动）
                    move_distance = current_distance * pull_factor * 0.3
                    new_src_x = src_module.x + (dx / distance) * move_distance * 0.5
                    new_src_y = src_module.y + (dy / distance) * move_distance * 0.5
                    new_dst_x = dst_module.x - (dx / distance) * move_distance * 0.5
                    new_dst_y = dst_module.y - (dy / distance) * move_distance * 0.5
                    
                    # 边界检查
                    new_src_x = max(0, min(self.layout_width - src_module.width, new_src_x))
                    new_src_y = max(0, min(self.layout_height - src_module.height, new_src_y))
                    new_dst_x = max(0, min(self.layout_width - dst_module.width, new_dst_x))
                    new_dst_y = max(0, min(self.layout_height - dst_module.height, new_dst_y))
                    
                    # 检查重叠
                    temp_src = src_module.copy(new_x=new_src_x, new_y=new_src_y)
                    temp_dst = dst_module.copy(new_x=new_dst_x, new_y=new_dst_y)
                    
                    # 检查是否与任何模块重叠
                    has_overlap = False
                    for other in modules:
                        if other.id == src_id or other.id == dst_id:
                            continue
                        if (self.check_module_overlap_simple(temp_src, other) or 
                            self.check_module_overlap_simple(temp_dst, other)):
                            has_overlap = True
                            break
                    
                    if not has_overlap:
                        src_module.x = new_src_x
                        src_module.y = new_src_y
                        dst_module.x = new_dst_x
                        dst_module.y = new_dst_y
                        improved = True
            
            if not improved:
                break
        
        final_hpwl = self._calculate_current_hpwl(modules)
        improvement = ((initial_hpwl - final_hpwl) / initial_hpwl * 100) if initial_hpwl > 0 else 0
        print(f"  连接图优化完成：HPWL从{initial_hpwl:.2f}降至{final_hpwl:.2f}（改进{improvement:.2f}%）")
        
        return modules
    
    def _iterative_hpwl_optimization(self, modules: List[PhysicalModule], max_rounds: int = 3) -> List[PhysicalModule]:
        """迭代式HPWL优化：多轮优化，每轮都尝试改进HPWL"""
        print(f"  迭代式HPWL优化（{max_rounds}轮）...")
        
        initial_hpwl = self._calculate_current_hpwl(modules)
        best_modules = [m.copy() for m in modules]
        best_hpwl = initial_hpwl
        
        for round_idx in range(max_rounds):
            # 计算连接权重
            connection_weights = self._calculate_connection_weights()
            
            # 按权重处理连接
            sorted_connections = sorted(connection_weights.items(), key=lambda x: x[1], reverse=True)
            
            module_map = {m.id: m for m in modules}
            
            for (src_id, dst_id), weight in sorted_connections:
                if weight <= 1:
                    continue
                
                src_module = module_map.get(src_id)
                dst_module = module_map.get(dst_id)
                
                if not src_module or not dst_module:
                    continue
                
                # 尝试小幅移动以降低HPWL
                src_center = src_module.get_center()
                dst_center = dst_module.get_center()
                
                # 计算当前HPWL贡献
                current_hpwl_contribution = abs(dst_center[0] - src_center[0]) + abs(dst_center[1] - src_center[1])
                
                # 尝试8个方向的微调
                best_move = None
                best_hpwl_improvement = 0
                
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                step_size = 2.0 * (1.0 - round_idx / max_rounds)  # 逐步减小步长
                
                for dx, dy in directions:
                    # 只移动源模块
                    new_src_x = src_module.x + dx * step_size
                    new_src_y = src_module.y + dy * step_size
                    
                    # 边界检查
                    if (0 <= new_src_x <= self.layout_width - src_module.width and
                        0 <= new_src_y <= self.layout_height - src_module.height):
                        
                        # 检查重叠
                        temp_src = src_module.copy(new_x=new_src_x, new_y=new_src_y)
                        has_overlap = False
                        for other in modules:
                            if other.id == src_id:
                                continue
                            if self.check_module_overlap_simple(temp_src, other):
                                has_overlap = True
                                break
                        
                        if not has_overlap:
                            # 计算新HPWL贡献
                            new_center = temp_src.get_center()
                            new_hpwl_contribution = abs(dst_center[0] - new_center[0]) + abs(dst_center[1] - new_center[1])
                            improvement = current_hpwl_contribution - new_hpwl_contribution
                            
                            if improvement > best_hpwl_improvement:
                                best_hpwl_improvement = improvement
                                best_move = (new_src_x, new_src_y)
                
                # 应用最佳移动
                if best_move:
                    src_module.x, src_module.y = best_move
            
            # 计算当前HPWL
            current_hpwl = self._calculate_current_hpwl(modules)
            if current_hpwl < best_hpwl:
                best_hpwl = current_hpwl
                best_modules = [m.copy() for m in modules]
        
        # 使用最佳结果
        final_hpwl = self._calculate_current_hpwl(best_modules)
        improvement = ((initial_hpwl - final_hpwl) / initial_hpwl * 100) if initial_hpwl > 0 else 0
        print(f"  迭代优化完成：HPWL从{initial_hpwl:.2f}降至{final_hpwl:.2f}（改进{improvement:.2f}%）")
        
        return best_modules
    
    def _gradient_based_hpwl_optimization(self, modules: List[PhysicalModule], max_iterations: int = 5) -> List[PhysicalModule]:
        """基于梯度下降的HPWL优化：计算HPWL对位置的梯度并沿梯度下降方向移动"""
        print(f"  梯度下降HPWL优化（{max_iterations}轮）...")
        
        initial_hpwl = self._calculate_current_hpwl(modules)
        module_map = {m.id: m for m in modules}
        
        # 构建连接图
        connections_by_module = defaultdict(list)
        for conn in self.connections:
            if len(conn) >= 4:
                src_id, dst_id = conn[0], conn[2]
                if src_id in module_map and dst_id in module_map:
                    connections_by_module[src_id].append(dst_id)
                    connections_by_module[dst_id].append(src_id)
        
        for iteration in range(max_iterations):
            improved = False
            learning_rate = 0.5 * (1.0 - iteration / max_iterations)  # 逐步减小学习率
            
            for module in modules:
                if module.type == 'obstacle' or getattr(module, 'is_fixed', False):
                    continue
                
                # 计算HPWL对模块位置的梯度（简化：使用有限差分）
                current_hpwl = self._calculate_current_hpwl(modules)
                
                # 计算x和y方向的梯度
                eps = 0.1
                best_dx, best_dy = 0, 0
                best_improvement = 0
                
                # 尝试4个方向的梯度
                for dx, dy in [(eps, 0), (-eps, 0), (0, eps), (0, -eps)]:
                    new_x = module.x + dx
                    new_y = module.y + dy
                    
                    # 边界检查
                    if (0 <= new_x <= self.layout_width - module.width and
                        0 <= new_y <= self.layout_height - module.height):
                        
                        # 检查重叠
                        temp_module = module.copy(new_x=new_x, new_y=new_y)
                        has_overlap = False
                        for other in modules:
                            if other.id == module.id:
                                continue
                            if self.check_module_overlap_simple(temp_module, other):
                                has_overlap = True
                                break
                        
                        if not has_overlap:
                            # 临时更新位置
                            old_x, old_y = module.x, module.y
                            module.x, module.y = new_x, new_y
                            
                            # 计算新HPWL
                            new_hpwl = self._calculate_current_hpwl(modules)
                            improvement = current_hpwl - new_hpwl
                            
                            # 恢复位置
                            module.x, module.y = old_x, old_y
                            
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_dx, best_dy = dx, dy
                
                # 沿最佳梯度方向移动
                if best_improvement > 0:
                    move_x = best_dx * learning_rate * 10  # 放大移动距离
                    move_y = best_dy * learning_rate * 10
                    
                    new_x = module.x + move_x
                    new_y = module.y + move_y
                    
                    # 边界检查
                    new_x = max(0, min(self.layout_width - module.width, new_x))
                    new_y = max(0, min(self.layout_height - module.height, new_y))
                    
                    # 最终重叠检查
                    temp_module = module.copy(new_x=new_x, new_y=new_y)
                    has_overlap = False
                    for other in modules:
                        if other.id == module.id:
                            continue
                        if self.check_module_overlap_simple(temp_module, other):
                            has_overlap = True
                            break
                    
                    if not has_overlap:
                        module.x = new_x
                        module.y = new_y
                        improved = True
            
            if not improved:
                break
        
        final_hpwl = self._calculate_current_hpwl(modules)
        improvement = ((initial_hpwl - final_hpwl) / initial_hpwl * 100) if initial_hpwl > 0 else 0
        print(f"  梯度下降优化完成：HPWL从{initial_hpwl:.2f}降至{final_hpwl:.2f}（改进{improvement:.2f}%）")
        
        return modules
    
    def _optimize_high_weight_connections(self, modules: List[PhysicalModule]) -> List[PhysicalModule]:
        """优化高权重连接对：专门处理最重要的连接"""
        print("  优化高权重连接对...")
        
        connection_weights = self._calculate_connection_weights()
        sorted_connections = sorted(connection_weights.items(), key=lambda x: x[1], reverse=True)
        
        module_map = {m.id: m for m in modules}
        initial_hpwl = self._calculate_current_hpwl(modules)
        
        # 只处理前20%的高权重连接
        top_connections = sorted_connections[:max(1, len(sorted_connections) // 5)]
        
        for (src_id, dst_id), weight in top_connections:
            src_module = module_map.get(src_id)
            dst_module = module_map.get(dst_id)
            
            if not src_module or not dst_module:
                continue
            
            # 计算当前距离
            src_center = src_module.get_center()
            dst_center = dst_module.get_center()
            current_distance = math.sqrt((dst_center[0] - src_center[0])**2 + 
                                        (dst_center[1] - src_center[1])**2)
            
            # 如果距离较大，尝试拉近
            if current_distance > 10.0:
                # 计算理想位置（中点）
                ideal_x = (src_center[0] + dst_center[0]) / 2
                ideal_y = (src_center[1] + dst_center[1]) / 2
                
                # 计算移动方向
                dx_src = ideal_x - src_center[0]
                dy_src = ideal_y - src_center[1]
                dx_dst = ideal_x - dst_center[0]
                dy_dst = ideal_y - dst_center[1]
                
                # 移动因子（保守）
                move_factor = 0.2
                
                # 计算新位置
                new_src_x = src_module.x + dx_src * move_factor
                new_src_y = src_module.y + dy_src * move_factor
                new_dst_x = dst_module.x + dx_dst * move_factor
                new_dst_y = dst_module.y + dy_dst * move_factor
                
                # 边界检查
                new_src_x = max(0, min(self.layout_width - src_module.width, new_src_x))
                new_src_y = max(0, min(self.layout_height - src_module.height, new_src_y))
                new_dst_x = max(0, min(self.layout_width - dst_module.width, new_dst_x))
                new_dst_y = max(0, min(self.layout_height - dst_module.height, new_dst_y))
                
                # 检查重叠
                temp_src = src_module.copy(new_x=new_src_x, new_y=new_src_y)
                temp_dst = dst_module.copy(new_x=new_dst_x, new_y=new_dst_y)
                
                has_overlap = False
                for other in modules:
                    if other.id == src_id or other.id == dst_id:
                        continue
                    if (self.check_module_overlap_simple(temp_src, other) or 
                        self.check_module_overlap_simple(temp_dst, other)):
                        has_overlap = True
                        break
                
                if not has_overlap:
                    src_module.x = new_src_x
                    src_module.y = new_src_y
                    dst_module.x = new_dst_x
                    dst_module.y = new_dst_y
        
        final_hpwl = self._calculate_current_hpwl(modules)
        improvement = ((initial_hpwl - final_hpwl) / initial_hpwl * 100) if initial_hpwl > 0 else 0
        print(f"  高权重连接优化完成：HPWL从{initial_hpwl:.2f}降至{final_hpwl:.2f}（改进{improvement:.2f}%）")
        
        return modules

    def _cluster_adjacent_modules(self, modules: List[PhysicalModule]) -> List[PhysicalModule]:
        """相邻模块聚类优化"""
        print("执行相邻模块聚类优化...")
        
        # 找出高连接度的模块
        module_connections = {}
        for module in modules:
            connections = sum(1 for conn in self.connections 
                           if conn[0] == module.id or conn[2] == module.id)
            module_connections[module.id] = connections
        
        # 按连接度排序
        sorted_modules = sorted(module_connections.items(), key=lambda x: x[1], reverse=True)
        
        # 将高连接度模块聚集到中心区域
        center_x = self.layout_width / 2
        center_y = self.layout_height / 2
        
        for i, (module_id, connections) in enumerate(sorted_modules[:5]):  # 前5个高连接度模块
            module = next((m for m in modules if m.id == module_id), None)
            if module and connections > 2:
                # 计算目标位置（围绕中心分布）
                angle = 2 * math.pi * i / 5
                target_x = center_x + 10 * math.cos(angle)
                target_y = center_y + 10 * math.sin(angle)
                
                # 平滑移动到目标位置
                module.x += (target_x - module.get_center()[0]) * 0.2
                module.y += (target_y - module.get_center()[1]) * 0.2
                
                # 边界约束
                module.x = max(0, min(self.layout_width - module.width, module.x))
                module.y = max(0, min(self.layout_height - module.height, module.y))
        
        return modules

    def _fine_tune_positions(self, modules: List[PhysicalModule]) -> List[PhysicalModule]:
        """精细位置调整：基于HPWL的局部搜索优化"""
        print("执行精细位置调整...")
        
        initial_hpwl = self._calculate_current_hpwl(modules)
        module_map = {m.id: m for m in modules}
        
        # 构建连接关系，优先优化有连接的模块
        connected_modules = set()
        for conn in self.connections:
            if len(conn) >= 4:
                src_id, dst_id = conn[0], conn[2]
                if src_id in module_map and dst_id in module_map:
                    connected_modules.add(src_id)
                    connected_modules.add(dst_id)
        
        # 先优化有连接的模块，再优化其他模块
        module_order = [m for m in modules if m.id in connected_modules] + \
                      [m for m in modules if m.id not in connected_modules]
        
        # 对每个模块进行微调
        for module in module_order:
            if module.type == 'obstacle' or getattr(module, 'is_fixed', False):
                continue
            
            best_position = None
            best_hpwl = float('inf')
            
            # 尝试多个方向和步长的微调
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            step_sizes = [1.0, 0.5, 0.25]  # 多级步长
            
            for step_size in step_sizes:
                for dx, dy in directions:
                    # 计算新位置
                    new_x = module.x + dx * step_size
                    new_y = module.y + dy * step_size
                    
                    # 边界检查
                    if (0 <= new_x <= self.layout_width - module.width and 
                        0 <= new_y <= self.layout_height - module.height):
                        
                        # 检查重叠
                        temp_module = module.copy(new_x=new_x, new_y=new_y)
                        has_overlap = False
                        for other in modules:
                            if other.id == module.id:
                                continue
                            if self.check_module_overlap_simple(temp_module, other):
                                has_overlap = True
                                break
                        
                        if not has_overlap:
                            # 临时更新位置
                            old_x, old_y = module.x, module.y
                            module.x, module.y = new_x, new_y
                            
                            # 计算HPWL
                            current_hpwl = self._calculate_current_hpwl(modules)
                            
                            if current_hpwl < best_hpwl:
                                best_hpwl = current_hpwl
                                best_position = (new_x, new_y)
                            
                            # 恢复原位置
                            module.x, module.y = old_x, old_y
            
            # 应用最佳位置
            if best_position:
                module.x, module.y = best_position
        
        final_hpwl = self._calculate_current_hpwl(modules)
        improvement = ((initial_hpwl - final_hpwl) / initial_hpwl * 100) if initial_hpwl > 0 else 0
        print(f"  精细调整完成：HPWL从{initial_hpwl:.2f}降至{final_hpwl:.2f}（改进{improvement:.2f}%）")
        
        return modules

    def _calculate_current_hpwl(self, modules: List[PhysicalModule]) -> float:
        """计算当前布局的HPWL（标准HPWL算法：网表最小包围盒）"""
        from collections import defaultdict
        
        total_hpwl = 0.0
        
        # 创建模块ID到模块的映射
        module_map = {module.id: module for module in modules}
        
        # 构建网表：将连接关系转换为网表
        nets = defaultdict(set)  # 使用set避免重复引脚
        
        for connection in self.connections:
            if len(connection) >= 2:
                # 处理两种连接格式
                if len(connection) == 4:
                    # 格式: (src_id, src_pin, dst_id, dst_pin)
                    module1_id, module2_id = connection[0], connection[2]
                elif len(connection) == 2:
                    # 格式: (src_id, dst_id)
                    module1_id, module2_id = connection[0], connection[1]
                else:
                    continue
                
                if module1_id in module_map and module2_id in module_map:
                    # 为每个连接创建唯一的网表ID
                    net_id = f"net_{module1_id}_{module2_id}"
                    nets[net_id].add(module1_id)
                    nets[net_id].add(module2_id)
        
        # 计算每个网表的HPWL
        for net_id, module_ids in nets.items():
            if len(module_ids) >= 2:
                # 收集网表中所有模块的引脚坐标
                pin_coords = []
                
                for module_id in module_ids:
                    module = module_map[module_id]
                    
                    # 获取模块的实际引脚坐标（相对于模块左下角）
                    for pin_rel_x, pin_rel_y in module.pins:
                        # 转换为全局坐标
                        pin_global_x = module.x + pin_rel_x
                        pin_global_y = module.y + pin_rel_y
                        pin_coords.append((pin_global_x, pin_global_y))
                
                if len(pin_coords) >= 2:
                    # 收集所有引脚的x和y坐标
                    x_coords = [pin[0] for pin in pin_coords]
                    y_coords = [pin[1] for pin in pin_coords]
                    
                    # 计算最小包围盒
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    
                    # 标准HPWL公式：HPWL = (Xmax - Xmin) + (Ymax - Ymin)
                    hpwl = (xmax - xmin) + (ymax - ymin)
                    total_hpwl += hpwl
        
        return total_hpwl

    def _apply_hpwl_optimization_to_positions(self, positions: Dict[str, Tuple[float, float]], 
                                            strategy: Dict) -> float:
        """对位置应用HPWL优化"""
        # 计算连接权重
        connection_weights = self._calculate_connection_weights()
        
        # 按权重排序连接
        sorted_connections = sorted(connection_weights.items(), key=lambda x: x[1], reverse=True)
        
        # 对高权重连接进行位置优化
        optimized_positions = positions.copy()
        for (src_id, dst_id), weight in sorted_connections:
            if weight > 1 and src_id in optimized_positions and dst_id in optimized_positions:
                # 计算当前距离
                src_pos = optimized_positions[src_id]
                dst_pos = optimized_positions[dst_id]
                current_distance = abs(dst_pos[0] - src_pos[0]) + abs(dst_pos[1] - src_pos[1])
                
                # 如果距离过大，尝试拉近
                if current_distance > 15.0:  # 距离阈值
                    # 计算中点
                    mid_x = (src_pos[0] + dst_pos[0]) / 2
                    mid_y = (src_pos[1] + dst_pos[1]) / 2
                    
                    # 移动因子
                    move_factor = strategy.get("attract_factor", 0.8) * 0.3
                    
                    # 计算新位置
                    new_src_x = src_pos[0] + (mid_x - src_pos[0]) * move_factor
                    new_src_y = src_pos[1] + (mid_y - src_pos[1]) * move_factor
                    new_dst_x = dst_pos[0] + (mid_x - dst_pos[0]) * move_factor
                    new_dst_y = dst_pos[1] + (mid_y - dst_pos[1]) * move_factor
                    
                    # 边界约束
                    src_module = self.modules[src_id]
                    dst_module = self.modules[dst_id]
                    
                    new_src_x = max(0, min(self.layout_width - src_module.width, new_src_x))
                    new_src_y = max(0, min(self.layout_height - src_module.height, new_src_y))
                    new_dst_x = max(0, min(self.layout_width - dst_module.width, new_dst_x))
                    new_dst_y = max(0, min(self.layout_height - dst_module.height, new_dst_y))
                    
                    # 更新位置
                    optimized_positions[src_id] = (new_src_x, new_src_y)
                    optimized_positions[dst_id] = (new_dst_x, new_dst_y)
        
        # 重新计算HPWL
        return self._calculate_layout_hpwl(optimized_positions)

    def _analyze_hpwl_optimization_effect(self, modules: List[PhysicalModule]) -> None:
        """分析HPWL优化效果"""
        from collections import defaultdict
        
        # 计算原始HPWL（使用标准HPWL算法）
        original_hpwl = 0.0
        
        # 构建网表：将连接关系转换为网表
        nets = defaultdict(set)  # 使用set避免重复引脚
        
        for connection in self.connections:
            if len(connection) >= 2:
                # 处理两种连接格式
                if len(connection) == 4:
                    # 格式: (src_id, src_pin, dst_id, dst_pin)
                    module1_id, module2_id = connection[0], connection[2]
                elif len(connection) == 2:
                    # 格式: (src_id, dst_id)
                    module1_id, module2_id = connection[0], connection[1]
                else:
                    continue
                
                if module1_id in self.original_positions and module2_id in self.original_positions:
                    # 为每个连接创建唯一的网表ID
                    net_id = f"net_{module1_id}_{module2_id}"
                    nets[net_id].add(module1_id)
                    nets[net_id].add(module2_id)
        
        # 计算每个网表的HPWL
        for net_id, module_ids in nets.items():
            if len(module_ids) >= 2:
                # 收集网表中所有模块的引脚坐标
                pin_coords = []
                
                for module_id in module_ids:
                    module = self.modules[module_id]
                    module_pos = self.original_positions[module_id]
                    
                    # 获取模块的实际引脚坐标（相对于模块左下角）
                    for pin_rel_x, pin_rel_y in module.pins:
                        # 转换为全局坐标
                        pin_global_x = module_pos[0] + pin_rel_x
                        pin_global_y = module_pos[1] + pin_rel_y
                        pin_coords.append((pin_global_x, pin_global_y))
                
                if len(pin_coords) >= 2:
                    # 收集所有引脚的x和y坐标
                    x_coords = [pin[0] for pin in pin_coords]
                    y_coords = [pin[1] for pin in pin_coords]
                    
                    # 计算最小包围盒
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    
                    # 标准HPWL公式：HPWL = (Xmax - Xmin) + (Ymax - Ymin)
                    hpwl = (xmax - xmin) + (ymax - ymin)
                    original_hpwl += hpwl
        
        # 计算优化后HPWL
        optimized_hpwl = self._calculate_current_hpwl(modules)
        
        # 计算改进百分比
        improvement = ((original_hpwl - optimized_hpwl) / original_hpwl) * 100 if original_hpwl > 0 else 0
        
        print(f"  HPWL优化效果分析:")
        print(f"    原始HPWL: {original_hpwl:.2f}")
        print(f"    优化后HPWL: {optimized_hpwl:.2f}")
        print(f"    HPWL改进: {improvement:.2f}%")
        
        # 分析连接距离分布
        connection_distances = []
        for src_id, _, dst_id, _ in self.connections:
            src_module = next((m for m in modules if m.id == src_id), None)
            dst_module = next((m for m in modules if m.id == dst_id), None)
            if src_module and dst_module:
                distance = abs(dst_module.get_center()[0] - src_module.get_center()[0]) + \
                          abs(dst_module.get_center()[1] - src_module.get_center()[1])
                connection_distances.append(distance)
        
        if connection_distances:
            avg_distance = sum(connection_distances) / len(connection_distances)
            max_distance = max(connection_distances)
            min_distance = min(connection_distances)
            
            print(f"    平均连接距离: {avg_distance:.2f}")
            print(f"    最大连接距离: {max_distance:.2f}")
            print(f"    最小连接距离: {min_distance:.2f}")
            
            # 分析距离分布
            short_connections = sum(1 for d in connection_distances if d < 10)
            medium_connections = sum(1 for d in connection_distances if 10 <= d < 30)
            long_connections = sum(1 for d in connection_distances if d >= 30)
            
            print(f"    短距离连接(<10): {short_connections} ({short_connections/len(connection_distances)*100:.1f}%)")
            print(f"    中距离连接(10-30): {medium_connections} ({medium_connections/len(connection_distances)*100:.1f}%)")
            print(f"    长距离连接(>=30): {long_connections} ({long_connections/len(connection_distances)*100:.1f}%)")

    def _generate_layout_with_strategy(self, agent: Dict, mst: nx.Graph) -> Tuple[
        Dict[str, Tuple[float, float]], float]:
        """使用特定策略生成布局，添加随机扰动增强探索"""
        strategy = agent["strategy"]

        try:
            # 设置随机种子以确保可重复但代理特定
            import random
            random.seed(strategy["seed"])
            np.random.seed(int(strategy["seed"]))

            # 使用nx.spring_layout生成布局（力导向）
            pos = nx.spring_layout(
                mst,
                weight='weight',
                k=strategy["k"],  # 变异后的理想距离
                iterations=strategy["iterations"],
                scale=1.0 / strategy["weight_factor"],
                seed=int(strategy["seed"])  # 传递种子到networkx
            )

            # 映射坐标到实际范围（考虑模块尺寸）
            all_modules = list(self.modules.values())
            if not all_modules:
                return {}, 0.0

            min_x = min(module.x for module in all_modules)
            max_x = max(module.x + module.width for module in all_modules)
            min_y = min(module.y for module in all_modules)
            max_y = max(module.y + module.height for module in all_modules)

            layout_width = max_x - min_x
            layout_height = max_y - min_y

            # 布局算法坐标范围归一化
            layout_positions = [pos.get(node, (0, 0)) for node in self.modules.keys() if node in pos]
            if not layout_positions:
                return {}, 0.0

            layout_min_x = min(x for x, y in layout_positions)
            layout_max_x = max(x for x, y in layout_positions)
            layout_min_y = min(y for x, y in layout_positions)
            layout_max_y = max(y for x, y in layout_positions)

            layout_range_x = max(layout_max_x - layout_min_x, 1e-10)
            layout_range_y = max(layout_max_y - layout_min_y, 1e-10)

            scaled_positions = {}
            for mod_id in self.modules:
                if mod_id in pos:
                    x, y = pos[mod_id]
                    # 归一化并映射
                    norm_x = (x - layout_min_x) / layout_range_x
                    norm_y = (y - layout_min_y) / layout_range_y
                    # 映射到实际范围，考虑尺寸
                    scaled_x = min_x + norm_x * (layout_width - self.modules[mod_id].width)
                    scaled_y = min_y + norm_y * (layout_height - self.modules[mod_id].height)
                    # 添加随机扰动：高斯噪声，增强探索（std=5% of range）
                    noise_std_x = 0.05 * layout_width
                    noise_std_y = 0.05 * layout_height
                    scaled_x += np.random.normal(0, noise_std_x)
                    scaled_y += np.random.normal(0, noise_std_y)
                    # 边界约束：防止扰动导致溢出
                    scaled_x = max(min_x, min(scaled_x, max_x - self.modules[mod_id].width))
                    scaled_y = max(min_y, min(scaled_y, max_y - self.modules[mod_id].height))
                    scaled_positions[mod_id] = (scaled_x, scaled_y)
                else:
                    # 如果模块不在MST中，使用原始位置 + 小扰动
                    orig_x, orig_y = self.original_positions[mod_id]
                    noise_std = 0.02 * max(layout_width, layout_height)  # 更小扰动
                    scaled_x = orig_x + np.random.normal(0, noise_std)
                    scaled_y = orig_y + np.random.normal(0, noise_std)
                    module = self.modules[mod_id]
                    scaled_x = max(min_x, min(scaled_x, max_x - module.width))
                    scaled_y = max(min_y, min(scaled_y, max_y - module.height))
                    scaled_positions[mod_id] = (scaled_x, scaled_y)

            # 计算扰动后的HPWL
            hpwl = self._calculate_layout_hpwl(scaled_positions)

            # 如果是HPWL优化策略，应用额外的HPWL优化
            if strategy["name"] in ["HPWLOptimizer", "ConnectionWeight"]:
                hpwl = self._apply_hpwl_optimization_to_positions(scaled_positions, strategy)

            return scaled_positions, hpwl

        except Exception as e:
            print(f"布局代理{agent['id']} 生成布局时出错: {e}")
            return {mod_id: self.original_positions[mod_id] for mod_id in self.modules}, float('inf')

    def optimize_layout(self, max_iterations: int = 10, num_runs_per_agent: int = 3) -> List[PhysicalModule]:
        """执行多智能体布局优化，以HPWL为唯一衡量指标，增强探索（多次运行+扰动 + 紧凑分布）"""
        print("开始构建连接图...")
        G = self.build_connectivity_graph()

        print("计算最小生成树...")
        mst = self.compute_mst(G)
        print(f"MST包含 {len(mst.nodes)} 个节点和 {len(mst.edges)} 条边")

        print("\n=== 多智能体布局竞赛开始（增强探索模式） ===")

        # 每个智能体生成布局（多次运行取最佳）
        best_layout = None
        best_hpwl = float('inf')
        best_agent = None

        agent_layouts = []
        for agent in self.layout_agents:
            print(f"\n智能体{agent['id']} ({agent['strategy']['name']}) 开始布局（{num_runs_per_agent}次运行）...")

            agent_best_hpwl = float('inf')
            agent_best_positions = None

            original_seed = agent['strategy']['seed']
            for run in range(num_runs_per_agent):
                # 每次运行使用代理种子 + run 偏移，确保变异
                run_seed = original_seed + run * 10
                agent['strategy']['seed'] = run_seed  # 临时设置

                # 生成布局
                positions, hpwl = self._generate_layout_with_strategy(agent, mst)

                if hpwl < agent_best_hpwl:
                    agent_best_hpwl = hpwl
                    agent_best_positions = positions

                # 恢复种子
                agent['strategy']['seed'] = original_seed

            # 记录代理最佳性能
            agent["hpwl_history"].append(agent_best_hpwl)
            agent["best_hpwl"] = min(agent["best_hpwl"], agent_best_hpwl)
            agent["best_layout"] = agent_best_positions if agent_best_hpwl < agent["best_hpwl"] else agent.get(
                "best_layout", agent_best_positions)

            agent_layouts.append((agent, agent_best_positions, agent_best_hpwl))

            print(f"  智能体{agent['id']} 最佳HPWL: {agent_best_hpwl:.2f} (来自{num_runs_per_agent}次运行)")

            # 更新全局最佳
            if agent_best_hpwl < best_hpwl:
                best_hpwl = agent_best_hpwl
                best_layout = agent_best_positions
                best_agent = agent
                print(f"  [冠军] 新全局最佳HPWL: {best_hpwl:.2f} (代理{agent['id']})")

        # 选择最佳布局
        print(f"\n=== 布局竞赛结果 ===")
        agent_layouts.sort(key=lambda x: x[2])  # 按最佳HPWL升序排序

        for rank, (agent, positions, hpwl) in enumerate(agent_layouts, 1):
            strategy_name = agent["strategy"]["name"]
            print(f"第{rank}名: 智能体{agent['id']} ({strategy_name}) - 最佳HPWL: {hpwl:.2f}")

        print(f"\n[冠军] 获胜策略: {best_agent['strategy']['name']} (HPWL: {best_hpwl:.2f})")

        # 使用最佳布局创建优化模块
        print("\n基于最佳布局调整模块位置...")
        optimized_modules = []
        for mod_id in self.modules:
            module = self.modules[mod_id]  # 在循环开始时定义module
            if mod_id in best_layout:
                x, y = best_layout[mod_id]
            else:
                x, y = self.original_positions[mod_id]

            optimized_modules.append(PhysicalModule(
                module_id=mod_id,
                x=x,
                y=y,
                width=module.width,
                height=module.height,
                module_type=module.type,
                layer=module.layer,
                pins=module.pins.copy(),
                spacing_guard_ratio=getattr(module, 'spacing_guard_ratio', 0.0),
                sensitivity=getattr(module, 'sensitivity', None)
            ))

        # ===== 阶段1：初始布局优化（边框激活） =====
        print("\n=== 阶段1：初始布局优化（边缘框激活，目标：边缘框间缝隙为0） ===")
        print("在此阶段，模块的边缘框（spacing guard）被视为模块的实际边界")
        print("目标：使所有边缘框之间的缝隙为0（不是模块本身之间的缝隙为0）")
        print("边缘框之间不允许重叠，确保初始布局满足间距要求")
        
        # 确保所有模块的边缘框都是激活状态，并设置边缘框大小为24%模块最短边
        for module in optimized_modules:
            module.spacing_guard_active = True
            module.spacing_guard_ratio = 0.24  # 边缘框大小为24%模块最短边
            module.spacing_guard_width = module._calculate_spacing_guard()  # 重新计算边缘框宽度
        print(f"已激活 {len(optimized_modules)} 个模块的边缘框（大小：24%模块最短边）")
        
        # 碰撞解决
        print("\n应用碰撞解决（包含边框检测）...")
        optimized_modules = self.resolve_collisions(optimized_modules)

        # 强制分离任何重叠（最高优先级）
        print("强制分离重叠模块（包含边框检测）...")
        optimized_modules = self.force_separate_overlapping_modules(optimized_modules, min_separation=0.000001)

        # 新增：完全消除边缘框间空隙缝隙减少，但禁止重叠
        print("完全消除边缘框间空隙缝隙减少（目标：边缘框间缝隙为0）...")
        optimized_modules = self.reduce_module_gaps(optimized_modules, target_gap=0.0)
        # 立即压缩守护边框间隙为0（多轮压缩确保收敛）
        for compress_round in range(3):
            optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=6)
            # 检查是否已收敛
            max_gap = 0.0
            for i in range(len(optimized_modules)):
                mod_i = optimized_modules[i]
                if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                    continue
                for j in range(i + 1, len(optimized_modules)):
                    mod_j = optimized_modules[j]
                    if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                        continue
                    gap = self.calculate_module_gap(mod_i, mod_j)
                    max_gap = max(max_gap, gap)
            if max_gap <= 1e-9:
                print(f"  压缩轮次 {compress_round+1}: 已收敛（最大边缘框缝隙={max_gap:.1e}um）")
                break

        # 再次强制分离（确保缝隙减少后无重叠）
        print("二次强制分离检查...")
        optimized_modules = self.force_separate_overlapping_modules(optimized_modules, min_separation=0.0)
        # 强制分离后立即压缩守护边框间隙为0
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=6)

        # 新增：超紧凑边缘框拉近，使边缘框分布极度紧凑但禁止重叠
        print("超紧凑边缘框分布优化（目标：边缘框间缝隙为0）...")
        optimized_modules = self.attract_modules(optimized_modules, attract_factor=0.99)
        # 拉近后立即压缩守护边框间隙为0（多轮压缩确保收敛）
        for compress_round in range(3):
            optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=6)
            # 检查是否已收敛
            max_gap = 0.0
            for i in range(len(optimized_modules)):
                mod_i = optimized_modules[i]
                if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                    continue
                for j in range(i + 1, len(optimized_modules)):
                    mod_j = optimized_modules[j]
                    if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                        continue
                    gap = self.calculate_module_gap(mod_i, mod_j)
                    max_gap = max(max_gap, gap)
            if max_gap <= 1e-9:
                print(f"  压缩轮次 {compress_round+1}: 已收敛（最大边缘框缝隙={max_gap:.1e}um）")
                break

        # 新增：完全消除边缘框间空隙渐进式间距压缩，进一步减小边缘框间距到完全为零
        print("完全消除边缘框间空隙渐进式间距压缩（目标：边缘框间缝隙为0）...")
        optimized_modules = self.progressive_gap_compression(optimized_modules, final_target_gap=0.0)
        # 渐进式压缩后立即压缩守护边框间隙为0（多轮压缩确保收敛）
        for compress_round in range(3):
            optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=6)
            # 检查是否已收敛
            max_gap = 0.0
            for i in range(len(optimized_modules)):
                mod_i = optimized_modules[i]
                if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                    continue
                for j in range(i + 1, len(optimized_modules)):
                    mod_j = optimized_modules[j]
                    if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                        continue
                    gap = self.calculate_module_gap(mod_i, mod_j)
                    max_gap = max(max_gap, gap)
            if max_gap <= 1e-9:
                print(f"  压缩轮次 {compress_round+1}: 已收敛（最大边缘框缝隙={max_gap:.1e}um）")
                break

        # 新增：完美边缘框边界接触优化，让边缘框完美边界接触
        print("完美边缘框边界接触优化（目标：边缘框间缝隙为0）...")
        optimized_modules = self.perfect_boundary_contact(optimized_modules, contact_tolerance=0.0)
        # 边界接触优化后立即压缩守护边框间隙为0（多轮压缩确保收敛）
        for compress_round in range(3):
            optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=6)
            # 检查是否已收敛
            max_gap = 0.0
            for i in range(len(optimized_modules)):
                mod_i = optimized_modules[i]
                if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                    continue
                for j in range(i + 1, len(optimized_modules)):
                    mod_j = optimized_modules[j]
                    if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                        continue
                    gap = self.calculate_module_gap(mod_i, mod_j)
                    max_gap = max(max_gap, gap)
            if max_gap <= 1e-9:
                print(f"  压缩轮次 {compress_round+1}: 已收敛（最大边缘框缝隙={max_gap:.1e}um）")
                break

        # 新增：完全消除边缘框间空隙微调优化，实现边缘框完美密度
        print("完全消除边缘框间空隙微调优化（目标：边缘框间缝隙为0）...")
        optimized_modules = self.micro_adjustment_optimization(optimized_modules, target_gap=0.0)
        # 微调优化后立即压缩守护边框间隙为0（多轮压缩确保收敛）
        for compress_round in range(3):
            optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=6)
            # 检查是否已收敛
            max_gap = 0.0
            for i in range(len(optimized_modules)):
                mod_i = optimized_modules[i]
                if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                    continue
                for j in range(i + 1, len(optimized_modules)):
                    mod_j = optimized_modules[j]
                    if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                        continue
                    gap = self.calculate_module_gap(mod_i, mod_j)
                    max_gap = max(max_gap, gap)
            if max_gap <= 1e-9:
                print(f"  压缩轮次 {compress_round+1}: 已收敛（最大边缘框缝隙={max_gap:.1e}um）")
                break

        # 新增：彻底消除所有边缘框缝隙（核心步骤：确保边缘框间缝隙为0）
        print("消除所有边缘框缝隙（核心步骤：确保边缘框间缝隙为0）...")
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=10)

        # 新增：分层HPWL优化，专门降低HPWL值
        print("执行分层HPWL优化...")
        optimized_modules = self._hierarchical_hpwl_optimization(optimized_modules)
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules)
        
        # 新增：基于连接图的HPWL优化（拉近高权重连接）
        print("执行基于连接图的HPWL优化...")
        optimized_modules = self._optimize_hpwl_by_connection_graph(optimized_modules, max_iterations=3)
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules)
        
        # 新增：迭代式HPWL优化（多轮优化）
        print("执行迭代式HPWL优化...")
        optimized_modules = self._iterative_hpwl_optimization(optimized_modules, max_rounds=3)
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules)
        
        # 新增：基于梯度下降的HPWL优化
        print("执行基于梯度下降的HPWL优化...")
        optimized_modules = self._gradient_based_hpwl_optimization(optimized_modules, max_iterations=5)
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules)
        
        # 新增：连接对优化（优化高权重连接对）
        print("执行连接对HPWL优化...")
        optimized_modules = self._optimize_high_weight_connections(optimized_modules)
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules)
        
        # 最终零缝隙压实（确保HPWL优化后仍保持边缘框间零缝隙）
        print("\n执行最终边缘框零缝隙压实（确保所有HPWL优化后仍保持边缘框间零缝隙）...")
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=10)

        # 最终零缝隙压实（最后一次确保边缘框间零缝隙）
        print("\n执行最终边缘框零缝隙压实（最后一次确保边缘框间零缝隙）...")
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=10)
        
        # 最终强制分离（确保完全消除空隙分布后无重叠）
        print("最终强制分离验证...")
        optimized_modules = self.force_separate_overlapping_modules(optimized_modules, min_separation=0.0)
        
        # 最终零缝隙压实（强制分离后再次压实边缘框）
        print("强制分离后再次边缘框零缝隙压实...")
        optimized_modules = self._enforce_zero_guard_gaps_complete(optimized_modules, tolerance=1e-9, max_rounds=8)

        # 边界约束
        optimized_modules = self.enforce_boundaries(optimized_modules)

        # 最终严格重叠验证和修复
        print("进行最终严格重叠验证...")
        optimized_modules = self.validate_and_fix_overlaps(optimized_modules, max_attempts=5)

        # 最终检查
        print("进行最终碰撞和间距检查...")

        # 分析边缘框间缝隙统计
        print("分析边缘框间缝隙统计（目标：边缘框间缝隙为0）...")
        gap_stats = self.analyze_gap_statistics(optimized_modules)

        # 新增：完美密度分析
        print("进行完美密度分析...")
        density_stats = self.perfect_density_analysis(optimized_modules)

        # 新增：HPWL优化效果分析
        print("进行HPWL优化效果分析...")
        self._analyze_hpwl_optimization_effect(optimized_modules)

        # 检查间距和重叠
        spacing_ok = self.check_spacing(optimized_modules)

        if spacing_ok:
            print("[成功] 最终检查通过，所有边缘框均已正确分离，无重叠")
            print(f"[成功] 阶段一完成：边缘框间平均缝隙={gap_stats['avg_gap']:.8f}um，最小缝隙={gap_stats['min_gap']:.8f}um")
            print(f"[重要] 注意：这是边缘框间的缝隙，不是模块本身之间的缝隙")
            print(f"[密度] 布局密度: {density_stats['density']:.8f} ({density_stats['density'] * 100:.6f}%)")
            print(f"[效率] 密度效率: {density_stats['efficiency'] * 100:.6f}%")
            print(f"[空隙] 空隙比例: {density_stats['gap_ratio'] * 100:.6f}%")
            print(f"[统计] 边缘框间距统计:")
            print(f"   平均间距: {density_stats['avg_gap']:.8f} um")
            print(f"   最小间距: {density_stats['min_gap']:.8f} um")
            print(f"   最大间距: {density_stats['max_gap']:.8f} um")
            print("[优化] 边缘框间距已压缩到接近零，实现边缘框完美布局密度")
            print("[完美] 阶段一完成：边缘框间零缝隙布局（边缘框保留，边缘框间缝隙为0）")
            print("[成功] 阶段一目标达成：边缘框保留，边缘框间缝隙为0！")
        else:
            print("[警告] 最终检查发现仍有问题")

        # ===== 阶段一完成检查点：验证状态并保存关键信息 =====
        print("\n" + "=" * 80)
        print("[阶段一检查点] 验证阶段一完成状态")
        print("=" * 80)
        
        # 验证所有模块的守护边框状态
        guard_active_count = sum(1 for m in optimized_modules 
                                 if getattr(m, 'spacing_guard_active', False))
        print(f"[验证] 守护边框激活状态: {guard_active_count}/{len(optimized_modules)} 个模块激活")
        
        if guard_active_count != len(optimized_modules):
            print(f"[警告] 有 {len(optimized_modules) - guard_active_count} 个模块的守护边框未激活，强制激活...")
            for module in optimized_modules:
                if not getattr(module, 'spacing_guard_active', False):
                    module.spacing_guard_active = True
        
        # 保存阶段一完成后的模块位置（用于后续验证）
        stage1_positions = {m.id: (m.x, m.y) for m in optimized_modules}
        stage1_guard_states = {m.id: getattr(m, 'spacing_guard_active', False) for m in optimized_modules}
        
        # 计算阶段一完成后的HPWL
        stage1_hpwl = self._calculate_current_hpwl(optimized_modules)
        print(f"[验证] 阶段一完成后的HPWL: {stage1_hpwl:.2f}")
        print(f"[验证] 阶段一完成后的模块数量: {len(optimized_modules)}")
        print(f"[验证] 阶段一完成后的模块位置已保存（用于阶段二验证）")
        print("=" * 80)

        # ===== 新增：分层后处理与可视化序列 =====
        try:
            print("\n生成首次可视化（含边缘框）: initial_with_guards.png")
            center_macro_vis = self.find_center_macro(optimized_modules)
            layer_info_vis = self.bfs_layering(center_macro_vis, optimized_modules)
            self.visualize_layer_progress(optimized_modules, layer_info_vis, filename="initial_with_guards.png", show_guards=True)
        except Exception as e:
            print(f"[可视化警告] 首次可视化失败: {e}")

        if not getattr(self, "enforce_zero_module_gaps", False):
            print("\n" + "=" * 80)
            print("[阶段二启动] 开始执行分层后处理：post_process_spacing_compression")
            print("=" * 80)
            print(f"[阶段二] 输入验证：接收 {len(optimized_modules)} 个来自阶段一的模块")
            print(f"[阶段二] 阶段一完成时的HPWL: {stage1_hpwl:.2f}")
            
            # 验证输入模块的状态
            input_guard_active = sum(1 for m in optimized_modules 
                                     if getattr(m, 'spacing_guard_active', False))
            print(f"[阶段二] 输入模块守护边框状态: {input_guard_active}/{len(optimized_modules)} 激活（预期全部激活）")
            
            # 验证输入模块位置是否与阶段一一致
            position_match_count = 0
            for module in optimized_modules:
                if module.id in stage1_positions:
                    orig_x, orig_y = stage1_positions[module.id]
                    if abs(module.x - orig_x) < 1e-9 and abs(module.y - orig_y) < 1e-9:
                        position_match_count += 1
            
            if position_match_count == len(optimized_modules):
                print(f"[阶段二] 输入验证通过：所有模块位置与阶段一一致")
            else:
                print(f"[阶段二警告] 有 {len(optimized_modules) - position_match_count} 个模块位置与阶段一不一致")
            
            # 调用阶段二处理
            final_modules = self.post_process_spacing_compression(optimized_modules)
            
            # 阶段二完成后的验证
            print("\n" + "=" * 80)
            print("[阶段二完成] 验证阶段二处理结果")
            print("=" * 80)
            
            # 验证阶段二是否改变了阶段一的位置（内层模块应该保持不变）
            stage2_positions = {m.id: (m.x, m.y) for m in final_modules}
            position_changes = []
            for mod_id, (stage1_x, stage1_y) in stage1_positions.items():
                if mod_id in stage2_positions:
                    stage2_x, stage2_y = stage2_positions[mod_id]
                    if abs(stage2_x - stage1_x) > 1e-6 or abs(stage2_y - stage1_y) > 1e-6:
                        position_changes.append((mod_id, stage1_x, stage1_y, stage2_x, stage2_y))
            
            if position_changes:
                print(f"[阶段二] 位置变化统计: {len(position_changes)} 个模块位置发生变化（这是正常的，阶段二会调整外层模块）")
                # 只显示前5个变化
                for mod_id, x1, y1, x2, y2 in position_changes[:5]:
                    dx, dy = x2 - x1, y2 - y1
                    print(f"  模块 {mod_id}: ({x1:.6f}, {y1:.6f}) -> ({x2:.6f}, {y2:.6f}) [Δx={dx:.6f}, Δy={dy:.6f}]")
            else:
                print(f"[阶段二] 所有模块位置保持不变（可能跳过了阶段二处理）")
            
            # 验证守护边框状态（阶段二应该废弃守护边框）
            stage2_guard_active = sum(1 for m in final_modules 
                                      if getattr(m, 'spacing_guard_active', False))
            print(f"[阶段二] 守护边框状态: {stage2_guard_active}/{len(final_modules)} 激活（预期全部废弃）")
            
            # 计算阶段二完成后的HPWL
            stage2_hpwl = self._calculate_current_hpwl(final_modules)
            hpwl_change = stage2_hpwl - stage1_hpwl
            print(f"[阶段二] 阶段二完成后的HPWL: {stage2_hpwl:.2f} (变化: {hpwl_change:+.2f})")
            print("=" * 80)
        else:
            print("\n[阶段2] 已启用零缝隙模式，跳过 post_process_spacing_compression，直接使用阶段1结果。")
            final_modules = optimized_modules

        try:
            print("\n生成最终可视化（去除边缘框）: final_layout.png")
            center_macro_final = self.find_center_macro(final_modules)
            layer_info_final = self.bfs_layering(center_macro_final, final_modules)
            self.visualize_layer_progress(final_modules, layer_info_final, filename="final_layout.png", show_guards=False)
        except Exception as e:
            print(f"[可视化警告] 最终可视化失败: {e}")

        print("多智能体布局优化完成（完全消除空隙分布）")
        return final_modules

    def build_connectivity_graph(self):
        """构建连接性图，权重表示连接重要性"""
        G = nx.Graph()

        # 添加所有模块作为节点
        for mod_id in self.modules:
            G.add_node(mod_id)

        # 添加连接作为边，权重基于连接类型和数量
        connection_counts = defaultdict(int)
        for src_id, _, dst_id, _ in self.connections:
            if src_id not in self.modules or dst_id not in self.modules:
                continue
            # 关键连接权重更高
            weight = 10.0 if any(self.modules[id].type in ['clk', 'analog']
                                 for id in [src_id, dst_id]) else 1.0
            key = (min(src_id, dst_id), max(src_id, dst_id))
            connection_counts[key] += weight

        # 添加边到图中
        for (src_id, dst_id), count in connection_counts.items():
            G.add_edge(src_id, dst_id, weight=1.0 / count)  # 连接越多，权重越小

        return G

    def compute_mst(self, G):
        """计算最小生成树"""
        return nx.minimum_spanning_tree(G, weight='weight')

    def check_module_collision(self, mod1, mod2):
        """检查两个模块是否碰撞（包括间距不足），支持吸引时的余量
        
        如果边框激活，则使用包含边框的边界进行检测
        """
        # 计算模块边界（如果边框激活，使用包含边框的边界）
        if getattr(mod1, 'spacing_guard_active', False):
            mod1_x1, mod1_x2, mod1_y1, mod1_y2 = mod1.get_bounds_with_guard()
        else:
            mod1_x1, mod1_x2, mod1_y1, mod1_y2 = mod1.get_bounds()
            
        if getattr(mod2, 'spacing_guard_active', False):
            mod2_x1, mod2_x2, mod2_y1, mod2_y2 = mod2.get_bounds_with_guard()
        else:
            mod2_x1, mod2_x2, mod2_y1, mod2_y2 = mod2.get_bounds()

        # 只在边框未激活时才额外添加间距要求
        if not (getattr(mod1, 'spacing_guard_active', False) or getattr(mod2, 'spacing_guard_active', False)):
            mod1_x1 -= self.min_spacing
            mod1_x2 += self.min_spacing
            mod1_y1 -= self.min_spacing
            mod1_y2 += self.min_spacing

            mod2_x1 -= self.min_spacing
            mod2_x2 += self.min_spacing
            mod2_y1 -= self.min_spacing
            mod2_y2 += self.min_spacing

        # 检查是否重叠
        x_overlap = not (mod1_x2 <= mod2_x1 or mod1_x1 >= mod2_x2)
        y_overlap = not (mod1_y2 <= mod2_y1 or mod1_y1 >= mod2_y2)

        return x_overlap and y_overlap

    def reduce_module_gaps(self, modules, target_gap: float = 0.0, max_iterations: int = 800):
        """完全消除空隙布局：将模块间距减少到完全为零，但严格禁止重叠"""
        import math

        print(f"完全消除空隙布局优化，目标间距: {target_gap}um（严格禁止重叠）...")

        module_list = modules.copy()

        # 渐进式目标间距：从当前平均间距逐步减少到完全为零
        current_gaps = []
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                gap = self.calculate_module_gap(module_list[i], module_list[j])
                if gap > 0:
                    current_gaps.append(gap)

        if current_gaps:
            avg_gap = sum(current_gaps) / len(current_gaps)
            print(f"当前平均间距: {avg_gap:.8f}um，目标间距: {target_gap}um")

            # 极激进的渐进式减少：分更多阶段逐步减少到完全为零
            stages = [avg_gap * 0.9, avg_gap * 0.7, avg_gap * 0.5, avg_gap * 0.3, avg_gap * 0.15, avg_gap * 0.08,
                      avg_gap * 0.04, avg_gap * 0.02, avg_gap * 0.01, avg_gap * 0.005, avg_gap * 0.001, 
                      avg_gap * 0.0005, avg_gap * 0.0001, target_gap]
            stages = [max(s, target_gap) for s in stages]  # 确保不小于目标间距
        else:
            stages = [target_gap]

        for stage_idx, stage_target in enumerate(stages):
            print(f"阶段 {stage_idx + 1}/{len(stages)}: 目标间距 {stage_target:.2f}um")

            for iteration in range(max_iterations // len(stages)):
                gaps_reduced = False

                # 检查所有模块对，尝试减少缝隙
                for i in range(len(module_list)):
                    mod_i = module_list[i]
                    if mod_i.type == 'obstacle':
                        continue

                    for j in range(i + 1, len(module_list)):
                        mod_j = module_list[j]
                        if mod_j.type == 'obstacle':
                            continue

                        # 计算当前间距
                        current_gap = self.calculate_module_gap(mod_i, mod_j)

                        # 如果间距大于阶段目标，尝试拉近
                        if current_gap > stage_target:
                            # 计算拉近方向
                            center_i = mod_i.get_center()
                            center_j = mod_j.get_center()
                            dx = center_j[0] - center_i[0]
                            dy = center_j[1] - center_i[1]
                            distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                            # 计算需要拉近的距离（更激进的拉近）
                            pull_distance = min(current_gap - stage_target, 5.0)  # 每次最多拉近5um，增强压缩能力

                            if pull_distance > 0.01:  # 更小的阈值，允许更精细的调整
                                # 计算拉近后的新位置
                                pull_factor = pull_distance / distance

                                # 计算新位置（向对方移动）
                                new_x_i = mod_i.x + dx * pull_factor * 0.5
                                new_y_i = mod_i.y + dy * pull_factor * 0.5
                                new_x_j = mod_j.x - dx * pull_factor * 0.5
                                new_y_j = mod_j.y - dy * pull_factor * 0.5

                                # 检查拉近后是否会产生重叠
                                temp_mod_i = mod_i.copy(new_x=new_x_i, new_y=new_y_i)
                                temp_mod_j = mod_j.copy(new_x=new_x_j, new_y=new_y_j)

                                # 严格检查：如果不会重叠，应用新位置
                                overlap, overlap_area = self.check_module_overlap(temp_mod_i, temp_mod_j)
                                if not overlap and overlap_area == 0.0:
                                    module_list[i] = temp_mod_i
                                    module_list[j] = temp_mod_j
                                    gaps_reduced = True

                # 如果当前阶段没有缝隙被减少，进入下一阶段
                if not gaps_reduced:
                    break

        # 最终统计
        final_gaps = []
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                gap = self.calculate_module_gap(module_list[i], module_list[j])
                if gap > 0:
                    final_gaps.append(gap)

        if final_gaps:
            final_avg = sum(final_gaps) / len(final_gaps)
            final_min = min(final_gaps)
            print(f"超紧凑完成：平均间距={final_avg:.2f}um，最小间距={final_min:.2f}um")

        return module_list

    def force_separate_overlapping_modules(self, modules, min_separation=0.0):
        """完全消除空隙强制分离：最小化分离距离到飞米级别，但确保无重叠"""
        import math

        print(f"完全消除空隙强制分离重叠模块（最小分离距离: {min_separation}um）...")

        module_list = modules.copy()
        max_iterations = 5000  # 大幅增加迭代次数以处理完全消除空隙情况
        separation_count = 0

        for iteration in range(max_iterations):
            has_overlaps = False

            for i in range(len(module_list)):
                mod_i = module_list[i]
                if mod_i.type == 'obstacle':
                    continue

                for j in range(i + 1, len(module_list)):
                    mod_j = module_list[j]
                    if mod_j.type == 'obstacle':
                        continue

                    # 检查重叠
                    overlap, overlap_area = self.check_module_overlap(mod_i, mod_j)

                    if overlap and overlap_area > 1e-12:  # 更严格的阈值
                        has_overlaps = True
                        separation_count += 1

                        # 计算分离方向
                        center_i = mod_i.get_center()
                        center_j = mod_j.get_center()
                        dx = center_j[0] - center_i[0]
                        dy = center_j[1] - center_i[1]
                        distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                        # 如果距离太小，使用随机方向
                        if distance < 1e-5:
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            distance = 1.0

                        # 计算最小分离距离（基于重叠面积）
                        # 使用更精确的分离距离计算
                        overlap_diag = np.sqrt(overlap_area)
                        required_distance = overlap_diag + min_separation

                        # 计算分离向量（更精确的分离）
                        separation_distance = max(required_distance, min_separation)
                        separation_x = (dx / distance) * separation_distance * 0.5
                        separation_y = (dy / distance) * separation_distance * 0.5

                        # 应用分离
                        new_x_i = mod_i.x - separation_x
                        new_y_i = mod_i.y - separation_y
                        new_x_j = mod_j.x + separation_x
                        new_y_j = mod_j.y + separation_y

                        # 创建分离后的模块
                        module_list[i] = mod_i.copy(new_x=new_x_i, new_y=new_y_i)
                        module_list[j] = mod_j.copy(new_x=new_x_j, new_y=new_y_j)

            # 如果没有重叠，提前结束
            if not has_overlaps:
                print(f"超紧凑分离完成，共处理 {separation_count} 个重叠，迭代 {iteration + 1} 次")
                break

        # 最终验证
        final_overlaps = 0
        total_overlap_area = 0.0
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                overlap, overlap_area = self.check_module_overlap(module_list[i], module_list[j])
                if overlap:
                    final_overlaps += 1
                    total_overlap_area += overlap_area

        if final_overlaps > 0:
            print(f"[警告] 仍有 {final_overlaps} 个重叠未解决，总重叠面积={total_overlap_area:.6f}")
        else:
            print("[成功] 所有重叠已成功分离（超紧凑模式）")

        return module_list

    def validate_and_fix_overlaps(self, modules, max_attempts=3):
        """验证并修复所有重叠，确保绝对无重叠"""
        print("开始严格重叠验证和修复...")

        module_list = modules.copy()

        for attempt in range(max_attempts):
            print(f"重叠验证尝试 {attempt + 1}/{max_attempts}")

            # 检查所有重叠
            overlaps_found = []
            for i in range(len(module_list)):
                for j in range(i + 1, len(module_list)):
                    overlap, overlap_area = self.check_module_overlap(module_list[i], module_list[j])
                    if overlap and overlap_area > 1e-12:  # 极小容差
                        overlaps_found.append((i, j, overlap_area))

            if not overlaps_found:
                print("[成功] 验证通过：无重叠发现")
                break

            print(f"发现 {len(overlaps_found)} 个重叠，开始修复...")

            # 按重叠面积排序，优先处理大面积重叠
            overlaps_found.sort(key=lambda x: x[2], reverse=True)

            # 修复重叠
            for i, j, overlap_area in overlaps_found:
                mod_i = module_list[i]
                mod_j = module_list[j]

                # 计算分离方向
                center_i = mod_i.get_center()
                center_j = mod_j.get_center()
                dx = center_j[0] - center_i[0]
                dy = center_j[1] - center_i[1]
                distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                # 如果距离太小，使用随机方向
                if distance < 1e-5:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    distance = 1.0

                # 计算分离距离（基于重叠面积）
                separation_distance = max(np.sqrt(overlap_area) + 1.0, 2.0)
                separation_x = (dx / distance) * separation_distance * 0.5
                separation_y = (dy / distance) * separation_distance * 0.5

                # 应用分离
                new_x_i = mod_i.x - separation_x
                new_y_i = mod_i.y - separation_y
                new_x_j = mod_j.x + separation_x
                new_y_j = mod_j.y + separation_y

                # 更新模块
                module_list[i] = mod_i.copy(new_x=new_x_i, new_y=new_y_i)
                module_list[j] = mod_j.copy(new_x=new_x_j, new_y=new_y_j)

        # 最终验证
        final_overlaps = 0
        total_overlap_area = 0.0
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                overlap, overlap_area = self.check_module_overlap(module_list[i], module_list[j])
                if overlap:
                    final_overlaps += 1
                    total_overlap_area += overlap_area

        if final_overlaps > 0:
            print(f"[失败] 验证失败：仍有 {final_overlaps} 个重叠，总重叠面积={total_overlap_area:.6f}")
        else:
            print("[成功] 验证成功：所有重叠已完全消除")

        return module_list

    def progressive_gap_compression(self, modules, final_target_gap=0.0, max_iterations_per_stage=80):
        """完全消除空隙渐进式间距压缩：逐步将模块间距压缩到完全为零"""
        print(f"开始完全消除空隙渐进式间距压缩，最终目标间距: {final_target_gap}um...")

        module_list = modules.copy()

        # 计算当前间距分布
        current_gaps = []
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                gap = self.calculate_module_gap(module_list[i], module_list[j])
                if gap > 0:
                    current_gaps.append(gap)

        if not current_gaps:
            print("没有找到有效间距，跳过压缩")
            return module_list

        current_avg = sum(current_gaps) / len(current_gaps)
        current_min = min(current_gaps)
        print(f"压缩前：平均间距={current_avg:.8f}um，最小间距={current_min:.8f}um")

        # 定义极激进的压缩阶段：从当前平均间距逐步压缩到完全为零（更细粒度）
        compression_stages = [
            current_avg * 0.8,  # 第一阶段：压缩到80%
            current_avg * 0.6,  # 第二阶段：压缩到60%
            current_avg * 0.4,  # 第三阶段：压缩到40%
            current_avg * 0.25,  # 第四阶段：压缩到25%
            current_avg * 0.15,  # 第五阶段：压缩到15%
            current_avg * 0.08,  # 第六阶段：压缩到8%
            current_avg * 0.04,  # 第七阶段：压缩到4%
            current_avg * 0.02,  # 第八阶段：压缩到2%
            current_avg * 0.01,  # 第九阶段：压缩到1%
            current_avg * 0.005,  # 第十阶段：压缩到0.5%
            current_avg * 0.002,  # 第十一阶段：压缩到0.2%
            current_avg * 0.001,  # 第十二阶段：压缩到0.1%
            current_avg * 0.0005,  # 第十三阶段：压缩到0.05%
            current_avg * 0.0001,  # 第十四阶段：压缩到0.01%
            current_avg * 0.00005,  # 第十五阶段：压缩到0.005%
            current_avg * 0.00001,  # 第十六阶段：压缩到0.001%
            final_target_gap  # 最终阶段：完全为零目标间距
        ]

        for stage_idx, target_gap in enumerate(compression_stages):
            print(f"压缩阶段 {stage_idx + 1}/{len(compression_stages)}: 目标间距 {target_gap:.2f}um")

            # 使用缝隙减少算法进行压缩（增加迭代次数）
            module_list = self.reduce_module_gaps(module_list, target_gap=target_gap, max_iterations=max_iterations_per_stage)

            # 强制分离确保无重叠
            module_list = self.force_separate_overlapping_modules(module_list, min_separation=0.1)

            # 统计当前阶段结果
            stage_gaps = []
            for i in range(len(module_list)):
                for j in range(i + 1, len(module_list)):
                    gap = self.calculate_module_gap(module_list[i], module_list[j])
                    if gap > 0:
                        stage_gaps.append(gap)

            if stage_gaps:
                stage_avg = sum(stage_gaps) / len(stage_gaps)
                stage_min = min(stage_gaps)
                print(f"阶段 {stage_idx + 1} 完成：平均间距={stage_avg:.2f}um，最小间距={stage_min:.2f}um")

        # 最终统计
        final_gaps = []
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                gap = self.calculate_module_gap(module_list[i], module_list[j])
                if gap > 0:
                    final_gaps.append(gap)

        if final_gaps:
            final_avg = sum(final_gaps) / len(final_gaps)
            final_min = min(final_gaps)
            compression_ratio = (current_avg - final_avg) / current_avg * 100
            print(f"渐进式压缩完成：平均间距={final_avg:.2f}um，最小间距={final_min:.2f}um")
            print(f"压缩效果：平均间距减少了 {compression_ratio:.1f}%")

        return module_list

    def micro_adjustment_optimization(self, modules, target_gap=0.0):
        """完全消除空隙微调优化：在飞米级别进行精细调整，实现完美密度"""
        print(f"开始完全消除空隙微调优化，目标间距: {target_gap}um...")

        module_list = modules.copy()
        max_iterations = 2000  # 增加迭代次数
        adjustment_count = 0

        for iteration in range(max_iterations):
            adjustments_made = 0

            # 对每个模块进行微调
            for i in range(len(module_list)):
                mod_i = module_list[i]
                if mod_i.type == 'obstacle':
                    continue

                # 计算与所有其他模块的间距
                min_gap = float('inf')
                closest_module = None

                for j in range(len(module_list)):
                    if i != j and module_list[j].type != 'obstacle':
                        gap = self.calculate_module_gap(mod_i, module_list[j])
                        if gap > 0 and gap < min_gap:
                            min_gap = gap
                            closest_module = module_list[j]

                # 如果间距大于目标间距，尝试微调
                if min_gap > target_gap and closest_module is not None:
                    # 计算微调方向（向最近模块移动）
                    center_i = mod_i.get_center()
                    center_j = closest_module.get_center()
                    dx = center_j[0] - center_i[0]
                    dy = center_j[1] - center_i[1]
                    distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                    # 计算微调距离（飞米级步长，但更积极）
                    micro_step = min(min_gap - target_gap, 0.0001)  # 每次最多微调0.0001um，增强压缩能力

                    if micro_step > 1e-12:  # 只有显著差距才微调
                        # 计算新位置
                        new_x = mod_i.x + (dx / distance) * micro_step * 0.5
                        new_y = mod_i.y + (dy / distance) * micro_step * 0.5

                        # 创建临时模块检查重叠
                        temp_module = mod_i.copy(new_x=new_x, new_y=new_y)

                        # 检查是否会产生重叠
                        has_overlap = False
                        for k, other_module in enumerate(module_list):
                            if k != i and other_module.type != 'obstacle':
                                overlap, _ = self.check_module_overlap(temp_module, other_module)
                                if overlap:
                                    has_overlap = True
                                    break

                        # 如果没有重叠，应用微调
                        if not has_overlap:
                            module_list[i] = temp_module
                            adjustments_made += 1
                            adjustment_count += 1

            # 如果没有微调，提前结束
            if adjustments_made == 0:
                print(f"微调优化完成，共进行 {adjustment_count} 次微调，迭代 {iteration + 1} 次")
                break

        # 最终统计
        final_gaps = []
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                gap = self.calculate_module_gap(module_list[i], module_list[j])
                if gap > 0:
                    final_gaps.append(gap)

        if final_gaps:
            final_avg = sum(final_gaps) / len(final_gaps)
            final_min = min(final_gaps)
            print(f"完全消除空隙微调优化完成：平均间距={final_avg:.8f}um，最小间距={final_min:.8f}um")

        return module_list

    def find_surrounding_neighbors(self, target_module: PhysicalModule, all_modules: List[PhysicalModule], 
                                   max_distance: float = None) -> Dict[str, List[Tuple[PhysicalModule, float]]]:
        """
        找到目标模块四周的相邻模块（上下左右四个方向）
        
        参数:
            target_module: 目标模块
            all_modules: 所有模块列表
            max_distance: 最大搜索距离（如果为None，则搜索所有模块）
        
        返回:
            Dict: {
                'left': [(module, gap), ...],    # 左侧相邻模块
                'right': [(module, gap), ...],   # 右侧相邻模块
                'top': [(module, gap), ...],     # 上方相邻模块
                'bottom': [(module, gap), ...]    # 下方相邻模块
            }
        """
        neighbors = {
            'left': [],
            'right': [],
            'top': [],
            'bottom': []
        }
        
        # 获取目标模块的边界（考虑guard）
        if getattr(target_module, 'spacing_guard_active', False):
            target_x1, target_x2, target_y1, target_y2 = target_module.get_bounds_with_guard()
        else:
            target_x1, target_x2, target_y1, target_y2 = target_module.get_bounds()
        
        target_center_x = (target_x1 + target_x2) / 2
        target_center_y = (target_y1 + target_y2) / 2
        
        for other_module in all_modules:
            if other_module.id == target_module.id:
                continue
            if other_module.type == 'obstacle':
                continue
            if not getattr(other_module, 'spacing_guard_active', False):
                continue
            
            # 获取其他模块的边界（考虑guard）
            if getattr(other_module, 'spacing_guard_active', False):
                other_x1, other_x2, other_y1, other_y2 = other_module.get_bounds_with_guard()
            else:
                other_x1, other_x2, other_y1, other_y2 = other_module.get_bounds()
            
            other_center_x = (other_x1 + other_x2) / 2
            other_center_y = (other_y1 + other_y2) / 2
            
            # 计算间隙
            gap = self.calculate_module_gap(target_module, other_module)
            
            # 如果重叠，跳过
            if gap < 0:
                continue
            
            # 如果距离太远，跳过
            if max_distance is not None and gap > max_distance:
                continue
            
            # 判断方向：检查在哪个方向相邻
            # 计算X和Y方向的重叠/间距
            x_overlap = min(target_x2, other_x2) - max(target_x1, other_x1)
            y_overlap = min(target_y2, other_y2) - max(target_y1, other_y1)
            
            # 左侧：其他模块在目标模块左侧，且Y方向有重叠
            if other_x2 <= target_x1 and y_overlap > -1e-6:
                neighbors['left'].append((other_module, gap))
            
            # 右侧：其他模块在目标模块右侧，且Y方向有重叠
            elif other_x1 >= target_x2 and y_overlap > -1e-6:
                neighbors['right'].append((other_module, gap))
            
            # 下方：其他模块在目标模块下方，且X方向有重叠
            elif other_y2 <= target_y1 and x_overlap > -1e-6:
                neighbors['bottom'].append((other_module, gap))
            
            # 上方：其他模块在目标模块上方，且X方向有重叠
            elif other_y1 >= target_y2 and x_overlap > -1e-6:
                neighbors['top'].append((other_module, gap))
        
        # 对每个方向的邻居按距离排序（最近的在前）
        for direction in neighbors:
            neighbors[direction].sort(key=lambda x: x[1])
        
        return neighbors

    def eliminate_spacing_guard_gaps(self, modules: List[PhysicalModule], tolerance: float = 1e-6,
                                     max_iterations: int = 1500) -> List[PhysicalModule]:
        """
        在边缘框激活阶段，彻底压缩边缘框之间的缝隙，让所有边缘框实现无缝接触。
        
        注意：目标是边缘框间的缝隙为0，不是模块本身之间的缝隙为0。
        
        增强版：优先处理四周相邻模块，确保压缩过程中不产生重叠。

        参数:
            modules: 当前模块列表
            tolerance: 允许的最大边缘框缝隙
            max_iterations: 最大迭代次数
        """
        active_modules = [m for m in modules if getattr(m, 'spacing_guard_active', False)]
        if len(active_modules) < 2:
            return modules

        def movable(module: PhysicalModule) -> bool:
            return module.type != 'obstacle' and not getattr(module, 'is_fixed', False)

        # 记录初始状态
        initial_gaps = []
        for i in range(len(modules)):
            mod_i = modules[i]
            if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                continue
            for j in range(i + 1, len(modules)):
                mod_j = modules[j]
                if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                    continue
                gap = self.calculate_module_gap(mod_i, mod_j)
                if gap > tolerance:
                    initial_gaps.append(gap)

        if not initial_gaps:
            return modules

        # 渐进式压缩：分阶段逐步压缩（更细粒度）
        stages = [0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.0]
        current_target = max(initial_gaps) if initial_gaps else tolerance
        
        for stage_idx, stage_factor in enumerate(stages):
            if stage_factor == 0.0:
                stage_target = tolerance
            else:
                stage_target = current_target * stage_factor
                stage_target = max(stage_target, tolerance)
            
            if stage_target < tolerance:
                stage_target = tolerance
            
            for iteration in range(max_iterations // len(stages)):
                # 策略1：优先处理四周相邻模块（更精确的压缩）
                surrounding_moves = 0
                for mod_i in modules:
                    if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                        continue
                    if not movable(mod_i):
                        continue
                    
                    # 找到四周相邻模块
                    neighbors = self.find_surrounding_neighbors(mod_i, modules, max_distance=current_target * 2)
                    
                    # 处理每个方向的最近邻居
                    for direction in ['left', 'right', 'top', 'bottom']:
                        if not neighbors[direction]:
                            continue
                        
                        # 优先处理最近的邻居
                        for neighbor_mod, gap in neighbors[direction][:2]:  # 每个方向最多处理2个最近的
                            if gap <= stage_target:
                                continue
                            
                            # 检查是否重叠
                            overlap, overlap_area = self.check_module_overlap(mod_i, neighbor_mod, tolerance=1e-9)
                            if overlap:
                                continue  # 已重叠，跳过
                            
                            # 尝试拉近
                            moved_module = self.move_module_closer(
                                mod_i,
                                neighbor_mod,
                                target_spacing=stage_target,
                                all_modules=modules
                            )
                            
                            if not moved_module:
                                continue
                            
                            # 验证移动后不重叠
                            has_overlap, overlapping_ids = self.check_overlap_with_all_modules(
                                moved_module, modules, exclude_ids={mod_i.id}
                            )
                            if has_overlap:
                                continue  # 移动后会产生重叠，放弃此次移动
                            
                            # 验证移动距离足够大
                            if (abs(moved_module.x - mod_i.x) <= 1e-12 and
                                    abs(moved_module.y - mod_i.y) <= 1e-12):
                                continue
                            
                            # 应用移动
                            mod_i.x = moved_module.x
                            mod_i.y = moved_module.y
                            surrounding_moves += 1
                            
                            # 每个模块每轮最多移动一次
                            break
                
                # 策略2：处理所有剩余的大缝隙（作为补充）
                gap_pairs: List[Tuple[float, PhysicalModule, PhysicalModule]] = []
                largest_gap = 0.0

                for i in range(len(modules)):
                    mod_i = modules[i]
                    if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                        continue

                    for j in range(i + 1, len(modules)):
                        mod_j = modules[j]
                        if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                            continue

                        gap = self.calculate_module_gap(mod_i, mod_j)
                        if gap > stage_target:
                            # 检查是否重叠
                            overlap, _ = self.check_module_overlap(mod_i, mod_j, tolerance=1e-9)
                            if not overlap:  # 只处理未重叠的
                                gap_pairs.append((gap, mod_i, mod_j))
                                largest_gap = max(largest_gap, gap)

                if gap_pairs:
                    gap_pairs.sort(key=lambda item: item[0], reverse=True)
                    moves_this_round = 0

                    # 每轮处理更多对（优先处理大缝隙）
                    for gap, mod_a, mod_b in gap_pairs[:min(200, len(gap_pairs))]:
                        if gap <= stage_target:
                            continue

                        # 选择可移动模块
                        mover, reference = None, None
                        if movable(mod_a):
                            mover, reference = mod_a, mod_b
                        elif movable(mod_b):
                            mover, reference = mod_b, mod_a
                        else:
                            continue  # 两者都不可移动

                        # 再次检查重叠（防止之前移动导致的新重叠）
                        overlap, _ = self.check_module_overlap(mover, reference, tolerance=1e-9)
                        if overlap:
                            continue

                        moved_module = self.move_module_closer(
                            mover,
                            reference,
                            target_spacing=stage_target,
                            all_modules=modules
                        )

                        if not moved_module:
                            continue

                        # 严格验证移动后不重叠
                        has_overlap, overlapping_ids = self.check_overlap_with_all_modules(
                            moved_module, modules, exclude_ids={mover.id}
                        )
                        if has_overlap:
                            continue  # 移动后会产生重叠，放弃此次移动

                        if (abs(moved_module.x - mover.x) <= 1e-12 and
                                abs(moved_module.y - mover.y) <= 1e-12):
                            continue

                        mover.x = moved_module.x
                        mover.y = moved_module.y
                        moves_this_round += 1

                # 如果两轮都没有移动，退出
                if surrounding_moves == 0 and (not gap_pairs or moves_this_round == 0):
                    break

                if largest_gap <= stage_target:
                    break

            # 更新当前目标
            if largest_gap > 0:
                current_target = largest_gap
            else:
                break

        # 最终验证和重叠修复
        print("  [验证] 检查压缩后的重叠情况...")
        overlaps_found = []
        for i in range(len(modules)):
            mod_i = modules[i]
            if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                continue
            for j in range(i + 1, len(modules)):
                mod_j = modules[j]
                if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                    continue
                overlap, overlap_area = self.check_module_overlap(mod_i, mod_j, tolerance=1e-9)
                if overlap and overlap_area > 1e-12:
                    overlaps_found.append((i, j, overlap_area))
        
        # 修复发现的重叠
        if overlaps_found:
            print(f"  [修复] 发现 {len(overlaps_found)} 个重叠，开始修复...")
            overlaps_found.sort(key=lambda x: x[2], reverse=True)  # 按重叠面积排序
            
            for idx_i, idx_j, overlap_area in overlaps_found:
                mod_i = modules[idx_i]
                mod_j = modules[idx_j]
                
                # 计算分离方向
                center_i = mod_i.get_center()
                center_j = mod_j.get_center()
                dx = center_j[0] - center_i[0]
                dy = center_j[1] - center_i[1]
                distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)
                
                # 如果距离太小，使用随机方向
                if distance < 1e-5:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    distance = 1.0
                
                # 计算分离距离（基于重叠面积）
                separation_distance = max(np.sqrt(overlap_area) + tolerance * 2, tolerance * 10)
                separation_x = (dx / distance) * separation_distance * 0.5
                separation_y = (dy / distance) * separation_distance * 0.5
                
                # 只移动可移动的模块
                if movable(mod_i) and not movable(mod_j):
                    # 只移动mod_i
                    new_x_i = mod_i.x - separation_x * 2
                    new_y_i = mod_i.y - separation_y * 2
                    temp_mod = mod_i.copy(new_x=new_x_i, new_y=new_y_i)
                    has_overlap, _ = self.check_overlap_with_all_modules(
                        temp_mod, modules, exclude_ids={mod_i.id}
                    )
                    if not has_overlap:
                        mod_i.x = new_x_i
                        mod_i.y = new_y_i
                elif movable(mod_j) and not movable(mod_i):
                    # 只移动mod_j
                    new_x_j = mod_j.x + separation_x * 2
                    new_y_j = mod_j.y + separation_y * 2
                    temp_mod = mod_j.copy(new_x=new_x_j, new_y=new_y_j)
                    has_overlap, _ = self.check_overlap_with_all_modules(
                        temp_mod, modules, exclude_ids={mod_j.id}
                    )
                    if not has_overlap:
                        mod_j.x = new_x_j
                        mod_j.y = new_y_j
                elif movable(mod_i) and movable(mod_j):
                    # 两者都可移动，同时移动
                    new_x_i = mod_i.x - separation_x
                    new_y_i = mod_i.y - separation_y
                    new_x_j = mod_j.x + separation_x
                    new_y_j = mod_j.y + separation_y
                    
                    temp_mod_i = mod_i.copy(new_x=new_x_i, new_y=new_y_i)
                    temp_mod_j = mod_j.copy(new_x=new_x_j, new_y=new_y_j)
                    
                    has_overlap_i, _ = self.check_overlap_with_all_modules(
                        temp_mod_i, modules, exclude_ids={mod_i.id, mod_j.id}
                    )
                    has_overlap_j, _ = self.check_overlap_with_all_modules(
                        temp_mod_j, modules, exclude_ids={mod_i.id, mod_j.id}
                    )
                    
                    if not has_overlap_i and not has_overlap_j:
                        mod_i.x = new_x_i
                        mod_i.y = new_y_i
                        mod_j.x = new_x_j
                        mod_j.y = new_y_j
            
            # 再次检查重叠
            final_overlaps = 0
            for i in range(len(modules)):
                mod_i = modules[i]
                if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                    continue
                for j in range(i + 1, len(modules)):
                    mod_j = modules[j]
                    if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                        continue
                    overlap, overlap_area = self.check_module_overlap(mod_i, mod_j, tolerance=1e-9)
                    if overlap and overlap_area > 1e-12:
                        final_overlaps += 1
            
            if final_overlaps > 0:
                print(f"  [警告] 仍有 {final_overlaps} 个重叠未完全修复")
            else:
                print(f"  [成功] 所有重叠已修复")
        
        # 最终统计缝隙
        final_gaps = []
        for i in range(len(modules)):
            mod_i = modules[i]
            if not getattr(mod_i, 'spacing_guard_active', False) or mod_i.type == 'obstacle':
                continue
            for j in range(i + 1, len(modules)):
                mod_j = modules[j]
                if not getattr(mod_j, 'spacing_guard_active', False) or mod_j.type == 'obstacle':
                    continue
                gap = self.calculate_module_gap(mod_i, mod_j)
                if gap > tolerance:
                    final_gaps.append(gap)

        if final_gaps:
            final_max = max(final_gaps)
            final_avg = sum(final_gaps) / len(final_gaps) if final_gaps else 0
            if final_max > tolerance * 10:  # 如果仍有较大缝隙，输出警告
                print(f"  [警告] 仍有 {len(final_gaps)} 对边缘框缝隙 > {tolerance:.1e}um，最大 {final_max:.6f}um，平均 {final_avg:.6f}um")
            else:
                print(f"  [信息] 剩余 {len(final_gaps)} 对边缘框缝隙，最大 {final_max:.6f}um，平均 {final_avg:.6f}um")
        else:
            print(f"  [成功] 所有边缘框缝隙已压缩至 <= {tolerance:.1e}um")

        return modules

    def snap_guard_pairs(self, modules: List[PhysicalModule], tolerance: float = 1e-9,
                         max_rounds: int = 6) -> List[PhysicalModule]:
        """
        进一步压实边缘框：无论是否在垂直方向重叠，均尝试让守护框边界贴合。

        该步骤专门处理 eliminate_spacing_guard_gaps 无法覆盖的"斜向缝隙"。
        """
        active_modules = [m for m in modules if getattr(m, 'spacing_guard_active', False)]
        if len(active_modules) < 2:
            return modules

        def movable(module: PhysicalModule) -> bool:
            return module.type != 'obstacle' and not getattr(module, 'is_fixed', False)

        # 收集所有需要处理的模块对
        gap_pairs = []
        for i in range(len(active_modules)):
            for j in range(i + 1, len(active_modules)):
                mod_a = active_modules[i]
                mod_b = active_modules[j]
                gap = self.calculate_module_gap(mod_a, mod_b)
                if gap > tolerance:
                    gap_pairs.append((gap, mod_a, mod_b))

        if not gap_pairs:
            return modules

        # 按缝隙大小排序，优先处理大缝隙
        gap_pairs.sort(key=lambda x: x[0], reverse=True)

        for round_idx in range(1, max_rounds + 1):
            adjustments = 0
            total_improvement = 0.0

            # 每轮处理所有对，但优先处理大缝隙
            for gap, mod_a, mod_b in gap_pairs:
                # 重新计算当前缝隙（可能已被之前的移动改变）
                current_gap = self.calculate_module_gap(mod_a, mod_b)
                if current_gap <= tolerance:
                    continue

                mover, reference = None, None
                if movable(mod_a):
                    mover, reference = mod_a, mod_b
                elif movable(mod_b):
                    mover, reference = mod_b, mod_a
                else:
                    continue

                moved = self.move_module_closer(
                    mover,
                    reference,
                    target_spacing=tolerance,
                    all_modules=modules
                )

                if moved and (abs(moved.x - mover.x) > 1e-12 or abs(moved.y - mover.y) > 1e-12):
                    improvement = current_gap - self.calculate_module_gap(moved, reference)
                    mover.x = moved.x
                    mover.y = moved.y
                    adjustments += 1
                    total_improvement += improvement

            if adjustments == 0:
                break

            # 更新gap_pairs（重新计算剩余缝隙）
            gap_pairs = []
            for i in range(len(active_modules)):
                for j in range(i + 1, len(active_modules)):
                    mod_a = active_modules[i]
                    mod_b = active_modules[j]
                    gap = self.calculate_module_gap(mod_a, mod_b)
                    if gap > tolerance:
                        gap_pairs.append((gap, mod_a, mod_b))
            
            gap_pairs.sort(key=lambda x: x[0], reverse=True)

            if not gap_pairs:
                break

        return modules

    def contact_optimization(self, modules, contact_tolerance=0.0001):
        """接触优化：让模块尽可能接触，但严格禁止重叠"""
        print(f"接触优化，接触容差: {contact_tolerance}um...")

        module_list = modules.copy()
        max_iterations = 300
        contact_count = 0

        for iteration in range(max_iterations):
            contacts_made = 0

            # 寻找可以接触的模块对
            for i in range(len(module_list)):
                mod_i = module_list[i]
                if mod_i.type == 'obstacle':
                    continue

                for j in range(i + 1, len(module_list)):
                    mod_j = module_list[j]
                    if mod_j.type == 'obstacle':
                        continue

                    # 计算当前间距
                    current_gap = self.calculate_module_gap(mod_i, mod_j)

                    # 如果间距在接触容差范围内，尝试让它们接触
                    if contact_tolerance < current_gap < contact_tolerance * 10:
                        # 计算接触方向
                        center_i = mod_i.get_center()
                        center_j = mod_j.get_center()
                        dx = center_j[0] - center_i[0]
                        dy = center_j[1] - center_i[1]
                        distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                        # 计算接触距离
                        contact_distance = current_gap - contact_tolerance

                        if contact_distance > 0.0001:  # 只有显著差距才调整
                            # 计算新位置（让模块接触）
                            new_x_i = mod_i.x + (dx / distance) * contact_distance * 0.5
                            new_y_i = mod_i.y + (dy / distance) * contact_distance * 0.5
                            new_x_j = mod_j.x - (dx / distance) * contact_distance * 0.5
                            new_y_j = mod_j.y - (dy / distance) * contact_distance * 0.5

                            # 检查接触后是否会产生重叠
                            temp_mod_i = mod_i.copy(new_x=new_x_i, new_y=new_y_i)
                            temp_mod_j = mod_j.copy(new_x=new_x_j, new_y=new_y_j)

                            # 严格检查：如果不会重叠，应用接触
                            overlap, overlap_area = self.check_module_overlap(temp_mod_i, temp_mod_j)
                            if not overlap and overlap_area == 0.0:
                                module_list[i] = temp_mod_i
                                module_list[j] = temp_mod_j
                                contacts_made += 1
                                contact_count += 1

            # 如果没有接触，提前结束
            if contacts_made == 0:
                print(f"接触优化完成，共建立 {contact_count} 个接触，迭代 {iteration + 1} 次")
                break

        # 最终统计
        final_gaps = []
        contact_pairs = 0
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                gap = self.calculate_module_gap(module_list[i], module_list[j])
                if gap > 0:
                    final_gaps.append(gap)
                    if gap <= contact_tolerance:
                        contact_pairs += 1

        if final_gaps:
            final_avg = sum(final_gaps) / len(final_gaps)
            final_min = min(final_gaps)
            print(f"接触优化完成：平均间距={final_avg:.4f}um，最小间距={final_min:.4f}um")
            print(f"接触模块对数量: {contact_pairs}")

        return module_list

    def density_analysis(self, modules):
        """密度分析：计算布局的理论最大密度"""
        print("进行密度分析...")

        # 计算所有模块的总面积
        total_module_area = sum(module.width * module.height for module in modules)

        # 计算布局边界框
        all_x = [module.x for module in modules]
        all_y = [module.y for module in modules]
        all_max_x = [module.x + module.width for module in modules]
        all_max_y = [module.y + module.height for module in modules]

        layout_width = max(all_max_x) - min(all_x)
        layout_height = max(all_max_y) - min(all_y)
        total_layout_area = layout_width * layout_height

        # 计算密度
        density = total_module_area / total_layout_area if total_layout_area > 0 else 0

        # 计算理论最大密度（假设模块可以完美填充）
        theoretical_max_density = 1.0  # 100%密度意味着无空隙

        print(f"布局密度分析:")
        print(f"  模块总面积: {total_module_area:.2f} um^2")
        print(f"  布局总面积: {total_layout_area:.2f} um^2")
        print(f"  当前密度: {density:.4f} ({density * 100:.2f}%)")
        print(f"  理论最大密度: {theoretical_max_density:.4f} ({theoretical_max_density * 100:.2f}%)")
        print(f"  密度效率: {density / theoretical_max_density * 100:.2f}%")

        return {
            'module_area': total_module_area,
            'layout_area': total_layout_area,
            'density': density,
            'theoretical_max_density': theoretical_max_density,
            'efficiency': density / theoretical_max_density
        }

    def perfect_contact_optimization(self, modules, contact_tolerance=0.0000001):
        """完美接触优化：让模块完美接触，但严格禁止重叠"""
        print(f"完美接触优化，接触容差: {contact_tolerance}um...")

        module_list = modules.copy()
        max_iterations = 500
        contact_count = 0

        for iteration in range(max_iterations):
            contacts_made = 0

            # 寻找可以完美接触的模块对
            for i in range(len(module_list)):
                mod_i = module_list[i]
                if mod_i.type == 'obstacle':
                    continue

                for j in range(i + 1, len(module_list)):
                    mod_j = module_list[j]
                    if mod_j.type == 'obstacle':
                        continue

                    # 计算当前间距
                    current_gap = self.calculate_module_gap(mod_i, mod_j)

                    # 如果间距在完美接触容差范围内，尝试让它们完美接触
                    if contact_tolerance < current_gap < contact_tolerance * 100:
                        # 计算接触方向
                        center_i = mod_i.get_center()
                        center_j = mod_j.get_center()
                        dx = center_j[0] - center_i[0]
                        dy = center_j[1] - center_i[1]
                        distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                        # 计算完美接触距离
                        perfect_contact_distance = current_gap - contact_tolerance

                        if perfect_contact_distance > 0.0000001:  # 只有显著差距才调整
                            # 计算新位置（让模块完美接触）
                            new_x_i = mod_i.x + (dx / distance) * perfect_contact_distance * 0.5
                            new_y_i = mod_i.y + (dy / distance) * perfect_contact_distance * 0.5
                            new_x_j = mod_j.x - (dx / distance) * perfect_contact_distance * 0.5
                            new_y_j = mod_j.y - (dy / distance) * perfect_contact_distance * 0.5

                            # 检查完美接触后是否会产生重叠
                            temp_mod_i = mod_i.copy(new_x=new_x_i, new_y=new_y_i)
                            temp_mod_j = mod_j.copy(new_x=new_x_j, new_y=new_y_j)

                            # 严格检查：如果不会重叠，应用完美接触
                            overlap, overlap_area = self.check_module_overlap(temp_mod_i, temp_mod_j)
                            if not overlap and overlap_area == 0.0:
                                module_list[i] = temp_mod_i
                                module_list[j] = temp_mod_j
                                contacts_made += 1
                                contact_count += 1

            # 如果没有接触，提前结束
            if contacts_made == 0:
                print(f"完美接触优化完成，共建立 {contact_count} 个完美接触，迭代 {iteration + 1} 次")
                break

        # 最终统计
        final_gaps = []
        perfect_contact_pairs = 0
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                gap = self.calculate_module_gap(module_list[i], module_list[j])
                if gap > 0:
                    final_gaps.append(gap)
                    if gap <= contact_tolerance:
                        perfect_contact_pairs += 1

        if final_gaps:
            final_avg = sum(final_gaps) / len(final_gaps)
            final_min = min(final_gaps)
            print(f"完美接触优化完成：平均间距={final_avg:.6f}um，最小间距={final_min:.6f}um")
            print(f"完美接触模块对数量: {perfect_contact_pairs}")

        return module_list

    def absolute_density_analysis(self, modules):
        """绝对密度分析：计算布局的绝对最大密度"""
        print("进行绝对密度分析...")

        # 计算所有模块的总面积
        total_module_area = sum(module.width * module.height for module in modules)

        # 计算布局边界框
        all_x = [module.x for module in modules]
        all_y = [module.y for module in modules]
        all_max_x = [module.x + module.width for module in modules]
        all_max_y = [module.y + module.height for module in modules]

        layout_width = max(all_max_x) - min(all_x)
        layout_height = max(all_max_y) - min(all_y)
        total_layout_area = layout_width * layout_height

        # 计算密度
        density = total_module_area / total_layout_area if total_layout_area > 0 else 0

        # 计算绝对最大密度（假设模块可以完美填充）
        absolute_max_density = 1.0  # 100%密度意味着绝对无空隙

        # 计算空隙面积
        gap_area = total_layout_area - total_module_area

        print(f"绝对密度分析:")
        print(f"  模块总面积: {total_module_area:.6f} um^2")
        print(f"  布局总面积: {total_layout_area:.6f} um^2")
        print(f"  空隙总面积: {gap_area:.6f} um^2")
        print(f"  当前密度: {density:.6f} ({density * 100:.4f}%)")
        print(f"  绝对最大密度: {absolute_max_density:.6f} ({absolute_max_density * 100:.4f}%)")
        print(f"  密度效率: {density / absolute_max_density * 100:.4f}%")
        print(f"  空隙比例: {gap_area / total_layout_area * 100:.4f}%")

        return {
            'module_area': total_module_area,
            'layout_area': total_layout_area,
            'gap_area': gap_area,
            'density': density,
            'absolute_max_density': absolute_max_density,
            'efficiency': density / absolute_max_density,
            'gap_ratio': gap_area / total_layout_area
        }

    def perfect_boundary_contact(self, modules, contact_tolerance=0.0):
        """完美边界接触：让模块完美接触边界，但严格禁止重叠"""
        print(f"完美边界接触优化，接触容差: {contact_tolerance}um...")

        module_list = modules.copy()
        max_iterations = 2000  # 增加迭代次数
        contact_count = 0

        for iteration in range(max_iterations):
            contacts_made = 0

            # 寻找可以完美边界接触的模块对
            for i in range(len(module_list)):
                mod_i = module_list[i]
                if mod_i.type == 'obstacle':
                    continue

                for j in range(i + 1, len(module_list)):
                    mod_j = module_list[j]
                    if mod_j.type == 'obstacle':
                        continue

                    # 计算当前间距
                    current_gap = self.calculate_module_gap(mod_i, mod_j)

                    # 如果间距大于目标间距，主动拉近（不仅处理已在容差范围内的）
                    if current_gap > contact_tolerance:
                        # 计算接触方向
                        center_i = mod_i.get_center()
                        center_j = mod_j.get_center()
                        dx = center_j[0] - center_i[0]
                        dy = center_j[1] - center_i[1]
                        distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                        # 计算完美边界接触距离（更积极的拉近）
                        perfect_boundary_distance = min(current_gap - contact_tolerance, 0.1)  # 每次最多拉近0.1um

                        if perfect_boundary_distance > 1e-12:  # 只有显著差距才调整
                            # 计算新位置（让模块完美边界接触）
                            new_x_i = mod_i.x + (dx / distance) * perfect_boundary_distance * 0.5
                            new_y_i = mod_i.y + (dy / distance) * perfect_boundary_distance * 0.5
                            new_x_j = mod_j.x - (dx / distance) * perfect_boundary_distance * 0.5
                            new_y_j = mod_j.y - (dy / distance) * perfect_boundary_distance * 0.5

                            # 检查完美边界接触后是否会产生重叠
                            temp_mod_i = mod_i.copy(new_x=new_x_i, new_y=new_y_i)
                            temp_mod_j = mod_j.copy(new_x=new_x_j, new_y=new_y_j)

                            # 严格检查：如果不会重叠，应用完美边界接触
                            overlap, overlap_area = self.check_module_overlap(temp_mod_i, temp_mod_j)
                            if not overlap and overlap_area == 0.0:
                                module_list[i] = temp_mod_i
                                module_list[j] = temp_mod_j
                                contacts_made += 1
                                contact_count += 1

            # 如果没有接触，提前结束
            if contacts_made == 0:
                print(f"完美边界接触优化完成，共建立 {contact_count} 个完美边界接触，迭代 {iteration + 1} 次")
                break

        # 最终统计
        final_gaps = []
        perfect_boundary_pairs = 0
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                gap = self.calculate_module_gap(module_list[i], module_list[j])
                if gap > 0:
                    final_gaps.append(gap)
                    if gap <= contact_tolerance + 1e-15:
                        perfect_boundary_pairs += 1

        if final_gaps:
            final_avg = sum(final_gaps) / len(final_gaps)
            final_min = min(final_gaps)
            print(f"完美边界接触优化完成：平均间距={final_avg:.8f}um，最小间距={final_min:.8f}um")
            print(f"完美边界接触模块对数量: {perfect_boundary_pairs}")

        return module_list

    def perfect_density_analysis(self, modules):
        """完美密度分析：计算布局的完美密度"""
        print("进行完美密度分析...")

        # 计算所有模块的总面积
        total_module_area = sum(module.width * module.height for module in modules)

        # 计算布局边界框
        all_x = [module.x for module in modules]
        all_y = [module.y for module in modules]
        all_max_x = [module.x + module.width for module in modules]
        all_max_y = [module.y + module.height for module in modules]

        layout_width = max(all_max_x) - min(all_x)
        layout_height = max(all_max_y) - min(all_y)
        total_layout_area = layout_width * layout_height

        # 计算密度
        density = total_module_area / total_layout_area if total_layout_area > 0 else 0

        # 计算完美密度（假设模块可以完美填充）
        perfect_density = 1.0  # 100%密度意味着完美无空隙

        # 计算空隙面积
        gap_area = total_layout_area - total_module_area

        # 计算模块间空隙
        module_gaps = []
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                gap = self.calculate_module_gap(modules[i], modules[j])
                if gap > 0:
                    module_gaps.append(gap)

        avg_gap = sum(module_gaps) / len(module_gaps) if module_gaps else 0
        min_gap = min(module_gaps) if module_gaps else 0
        max_gap = max(module_gaps) if module_gaps else 0

        print(f"完美密度分析:")
        print(f"  模块总面积: {total_module_area:.8f} um^2")
        print(f"  布局总面积: {total_layout_area:.8f} um^2")
        print(f"  空隙总面积: {gap_area:.8f} um^2")
        print(f"  当前密度: {density:.8f} ({density * 100:.6f}%)")
        print(f"  完美密度: {perfect_density:.8f} ({perfect_density * 100:.6f}%)")
        print(f"  密度效率: {density / perfect_density * 100:.6f}%")
        print(f"  空隙比例: {gap_area / total_layout_area * 100:.6f}%")
        print(f"  模块间距统计:")
        print(f"    平均间距: {avg_gap:.8f} um")
        print(f"    最小间距: {min_gap:.8f} um")
        print(f"    最大间距: {max_gap:.8f} um")

        return {
            'module_area': total_module_area,
            'layout_area': total_layout_area,
            'gap_area': gap_area,
            'density': density,
            'perfect_density': perfect_density,
            'efficiency': density / perfect_density,
            'gap_ratio': gap_area / total_layout_area,
            'avg_gap': avg_gap,
            'min_gap': min_gap,
            'max_gap': max_gap
        }

    def calculate_module_gap(self, mod1, mod2):
        """计算两个模块之间的最小间距
        
        如果边框激活，则使用包含边框的边界进行计算
        """
        # 计算模块边界（如果边框激活，使用包含边框的边界）
        if getattr(mod1, 'spacing_guard_active', False):
            mod1_x1, mod1_x2, mod1_y1, mod1_y2 = mod1.get_bounds_with_guard()
        else:
            mod1_x1, mod1_x2, mod1_y1, mod1_y2 = mod1.get_bounds()
            
        if getattr(mod2, 'spacing_guard_active', False):
            mod2_x1, mod2_x2, mod2_y1, mod2_y2 = mod2.get_bounds_with_guard()
        else:
            mod2_x1, mod2_x2, mod2_y1, mod2_y2 = mod2.get_bounds()

        # 计算X和Y方向的间距
        x_gap = min(mod1_x2, mod2_x2) - max(mod1_x1, mod2_x1)
        y_gap = min(mod1_y2, mod2_y2) - max(mod1_y1, mod2_y1)

        # 如果重叠，返回负值
        if x_gap > 0 and y_gap > 0:
            return -min(x_gap, y_gap)  # 重叠深度

        # 如果不重叠，返回最小间距
        if x_gap > 0:  # Y方向有间距
            return y_gap
        elif y_gap > 0:  # X方向有间距
            return x_gap
        else:  # 对角间距
            return np.sqrt(x_gap ** 2 + y_gap ** 2)

    def check_module_overlap(self, mod1, mod2, tolerance=1e-6):
        """严格检查两个模块是否重叠（不考虑间距），使用高精度检测
        
        如果边框激活，则使用包含边框的边界进行检测
        """
        # 获取模块边界（如果边框激活，使用包含边框的边界）
        if getattr(mod1, 'spacing_guard_active', False):
            mod1_x1, mod1_x2, mod1_y1, mod1_y2 = mod1.get_bounds_with_guard()
        else:
            mod1_x1, mod1_x2, mod1_y1, mod1_y2 = mod1.get_bounds()
            
        if getattr(mod2, 'spacing_guard_active', False):
            mod2_x1, mod2_x2, mod2_y1, mod2_y2 = mod2.get_bounds_with_guard()
        else:
            mod2_x1, mod2_x2, mod2_y1, mod2_y2 = mod2.get_bounds()

        # 高精度重叠检测：考虑浮点数精度问题
        x_overlap = not (mod1_x2 <= mod2_x1 + tolerance or mod1_x1 >= mod2_x2 - tolerance)
        y_overlap = not (mod1_y2 <= mod2_y1 + tolerance or mod1_y1 >= mod2_y2 - tolerance)

        # 如果重叠，计算重叠面积
        if x_overlap and y_overlap:
            overlap_x = min(mod1_x2, mod2_x2) - max(mod1_x1, mod2_x1)
            overlap_y = min(mod1_y2, mod2_y2) - max(mod1_y1, mod2_y1)
            overlap_area = max(0, overlap_x * overlap_y)
            return True, overlap_area
        else:
            return False, 0.0

    def check_module_overlap_simple(self, mod1, mod2):
        """简化版重叠检查，返回布尔值"""
        overlap, _ = self.check_module_overlap(mod1, mod2)
        return overlap

    def resolve_collisions(self, modules, max_iterations: int = 100):
        """解决模块之间的碰撞，确保有足够间距，加入随机方向增强探索"""
        import math
        import random

        print(f"解决模块碰撞并确保最小间距 {self.min_spacing}um（带随机方向）...")

        # 转换为列表以便修改
        module_list = modules.copy()
        module_dict = {m.id: m for m in module_list}

        # 记录每个模块的移动向量
        move_vectors = {m.id: (0, 0) for m in module_list}

        for iteration in range(max_iterations):
            has_collisions = False

            # 检查所有模块对
            for i in range(len(module_list)):
                mod_i = module_list[i]

                for j in range(i + 1, len(module_list)):
                    mod_j = module_list[j]

                    # 检查两个模块是否碰撞或间距不足
                    if self.check_module_collision(mod_i, mod_j):
                        has_collisions = True

                        # 计算模块边界
                        mod1_x1, mod1_x2, mod1_y1, mod1_y2 = mod_i.get_bounds()
                        mod2_x1, mod2_x2, mod2_y1, mod2_y2 = mod_j.get_bounds()

                        # 计算中心点
                        center_i = mod_i.get_center()
                        center_j = mod_j.get_center()

                        # 计算分离方向
                        dx = center_j[0] - center_i[0]
                        dy = center_j[1] - center_i[1]
                        distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                        # 添加随机方向偏移：±30度，增强探索性
                        angle_offset = random.uniform(-30, 30) * math.pi / 180  # 转换为弧度
                        # 旋转方向向量
                        cos_a, sin_a = math.cos(angle_offset), math.sin(angle_offset)
                        dx_rot = dx * cos_a - dy * sin_a
                        dy_rot = dx * sin_a + dy * cos_a
                        # 归一化旋转后的方向
                        dir_magnitude = np.sqrt(dx_rot ** 2 + dy_rot ** 2)
                        if dir_magnitude > 0:
                            dx_rot /= dir_magnitude
                            dy_rot /= dir_magnitude
                        else:
                            dx_rot, dy_rot = dx / distance, dy / distance

                        # 计算需要移动的距离（确保有最小间距）
                        min_distance = (np.sqrt(mod_i.width ** 2 + mod_i.height ** 2) +
                                        np.sqrt(mod_j.width ** 2 + mod_j.height ** 2)) / 2 + self.min_spacing
                        move_distance = max(0, min_distance - distance)

                        # 计算推开方向和距离（使用旋转方向）
                        if move_distance > 0:
                            # 根据模块类型和当前移动情况决定移动方向
                            push_x = dx_rot * move_distance * 0.5
                            push_y = dy_rot * move_distance * 0.5

                            # 更新移动向量
                            if mod_i.type != 'obstacle':
                                move_vectors[mod_i.id] = (
                                    move_vectors[mod_i.id][0] - push_x,
                                    move_vectors[mod_i.id][1] - push_y
                                )

                            if mod_j.type != 'obstacle':
                                move_vectors[mod_j.id] = (
                                    move_vectors[mod_j.id][0] + push_x,
                                    move_vectors[mod_j.id][1] + push_y
                                )

            # 应用移动向量
            for i, module in enumerate(module_list):
                if module.type != 'obstacle':
                    dx, dy = move_vectors[module.id]
                    new_x = module.x + dx
                    new_y = module.y + dy
                    module_list[i] = module.copy(new_x=new_x, new_y=new_y)

            # 重置移动向量
            move_vectors = {m.id: (0, 0) for m in module_list}

            # 如果没有碰撞和间距问题，提前结束
            if not has_collisions:
                print(f"碰撞和间距解决完成，共迭代 {iteration + 1} 次")
                break

            # 最后一次迭代后，如果仍有碰撞，使用更强制的方法（也添加随机）
            if iteration == max_iterations - 1 and has_collisions:
                print("警告：达到最大迭代次数，仍有碰撞或间距问题，使用强制分离方法（带随机）")
                # 强制分离所有模块
                for i in range(len(module_list)):
                    mod_i = module_list[i]
                    if mod_i.type == 'obstacle':
                        continue

                    for j in range(i + 1, len(module_list)):
                        mod_j = module_list[j]

                        if self.check_module_collision(mod_i, mod_j):
                            # 强制分离 - 直接移动到有足够间距的位置，添加随机偏移
                            mod1_x1, mod1_x2, mod1_y1, mod1_y2 = mod_i.get_bounds()
                            mod2_x1, mod2_x2, mod2_y1, mod2_y2 = mod_j.get_bounds()

                            # 计算中心点
                            center_i = mod_i.get_center()
                            center_j = mod_j.get_center()

                            # 计算分离方向 + 随机偏移
                            dx = center_j[0] - center_i[0]
                            dy = center_j[1] - center_i[1]
                            distance = max(np.sqrt(dx ** 2 + dy ** 2), 1e-5)

                            angle_offset = random.uniform(-30, 30) * math.pi / 180
                            cos_a, sin_a = math.cos(angle_offset), math.sin(angle_offset)
                            dx_rot = dx * cos_a - dy * sin_a
                            dy_rot = dx * sin_a + dy * cos_a
                            dir_magnitude = np.sqrt(dx_rot ** 2 + dy_rot ** 2)
                            if dir_magnitude > 0:
                                dx_rot /= dir_magnitude
                                dy_rot /= dir_magnitude

                            # 计算需要移动的距离
                            min_distance = (np.sqrt(mod_i.width ** 2 + mod_i.height ** 2) +
                                            np.sqrt(mod_j.width ** 2 + mod_j.height ** 2)) / 2 + self.min_spacing
                            move_distance = min_distance - distance + 1.0  # 额外1um安全余量

                            if move_distance > 0:
                                # 移动非障碍物模块
                                if mod_j.type != 'obstacle':
                                    new_x_j = mod_j.x + dx_rot * move_distance
                                    new_y_j = mod_j.y + dy_rot * move_distance
                                    module_list[j] = mod_j.copy(new_x=new_x_j, new_y=new_y_j)

        return module_list

    def enforce_boundaries(self, modules):
        """确保所有模块都在设计边界内"""
        # 计算设计边界（所有模块的合并边界）
        all_x = [module.x for module in modules]
        all_y = [module.y for module in modules]
        all_max_x = [module.x + module.width for module in modules]
        all_max_y = [module.y + module.height for module in modules]

        design_min_x = min(all_x)
        design_max_x = max(all_max_x)
        design_min_y = min(all_y)
        design_max_y = max(all_max_y)

        design_width = design_max_x - design_min_x
        design_height = design_max_y - design_min_y

        # 为目标边界添加一些边距
        target_min_x = design_min_x - design_width * 0.1
        target_max_x = design_max_x + design_width * 0.1
        target_min_y = design_min_y - design_height * 0.1
        target_max_y = design_max_y + design_height * 0.1

        # 调整超出边界的模块
        adjusted_modules = []
        for module in modules:
            new_x = module.x
            new_y = module.y

            # 检查并调整X边界
            if new_x < target_min_x:
                new_x = target_min_x
            elif new_x + module.width > target_max_x:
                new_x = target_max_x - module.width

            # 检查并调整Y边界
            if new_y < target_min_y:
                new_y = target_min_y
            elif new_y + module.height > target_max_y:
                new_y = target_max_y - module.height

            # 创建调整后的模块
            adjusted_module = module.copy(new_x=new_x, new_y=new_y)
            adjusted_modules.append(adjusted_module)

        return adjusted_modules

    def visualize_layout_comparison(self, original_modules,
                                    optimized_modules):
        """可视化布局优化前后的对比"""
        # 创建统一结果文件夹
        results_dir = "0_layout/code/layout_result"
        os.makedirs(results_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 计算坐标范围
        all_x = []
        all_y = []
        for module in original_modules + optimized_modules:
            all_x.extend([module.x, module.x + module.width])
            all_y.extend([module.y, module.y + module.height])

        x_margin = (max(all_x) - min(all_x)) * 0.1
        y_margin = (max(all_y) - min(all_y)) * 0.1
        x_min, x_max = min(all_x) - x_margin, max(all_x) + x_margin
        y_min, y_max = min(all_y) - y_margin, max(all_y) + y_margin

        # 绘制原始布局
        print("绘制原始布局...")
        for module in original_modules:
            color = self._get_module_color(module.type)

            rect = patches.Rectangle(
                (module.x, module.y), module.width, module.height,
                linewidth=2, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax1.add_patch(rect)
            ax1.text(module.x + module.width / 2, module.y + module.height / 2,
                     module.id, fontsize=8, ha='center', va='center')

        ax1.set_title("原始布局")
        ax1.set_xlabel("X坐标 (um)")
        ax1.set_ylabel("Y坐标 (um)")
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_aspect('equal')

        # 绘制优化后布局
        print("绘制优化后布局...")
        for module in optimized_modules:
            color = self._get_module_color(module.type)

            rect = patches.Rectangle(
                (module.x, module.y), module.width, module.height,
                linewidth=2, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax2.add_patch(rect)
            ax2.text(module.x + module.width / 2, module.y + module.height / 2,
                     module.id, fontsize=8, ha='center', va='center')

        ax2.set_title("完全消除空隙优化后布局")
        ax2.set_xlabel("X坐标 (um)")
        ax2.set_ylabel("Y坐标 (um)")
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_aspect('equal')

        # 添加图例
        self._add_legend(ax1)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "layout_optimization_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"布局对比图已保存到 {results_dir}/layout_optimization_comparison.png")

    def visualize_competition_results(self):
        """可视化多智能体布局竞赛结果：HPWL排名和趋势"""
        # 创建统一结果文件夹
        results_dir = "0_layout/code/layout_result"
        os.makedirs(results_dir, exist_ok=True)
        
        if not self.layout_agents:
            print("无布局智能体数据，无法可视化竞赛结果")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("多智能体布局竞赛结果", fontsize=14)
        
        # 1. HPWL排名条形图
        agent_ids = [agent['id'] for agent in self.layout_agents]
        agent_names = [agent['strategy']['name'] for agent in self.layout_agents]
        hpwls = [agent['best_hpwl'] for agent in self.layout_agents]
        
        # 排序以显示排名
        sorted_indices = sorted(range(len(hpwls)), key=lambda k: hpwls[k])
        sorted_hpwls = [hpwls[i] for i in sorted_indices]
        sorted_names = [agent_names[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_hpwls)))
        bars = ax1.bar(range(len(sorted_hpwls)), sorted_hpwls, color=colors)
        ax1.set_xlabel("排名")
        ax1.set_ylabel("HPWL")
        ax1.set_title("智能体HPWL排名")
        ax1.set_xticks(range(len(sorted_hpwls)))
        ax1.set_xticklabels([f"{name}\n(代理{i+1})" for i, name in enumerate(sorted_names)], rotation=45, ha='right')
        
        # 添加数值标签
        for bar, hpwl in zip(bars, sorted_hpwls):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sorted_hpwls)*0.01,
                     f"{hpwl:.2f}", ha='center', va='bottom', fontsize=9)
        
        # 2. HPWL趋势图（如果有历史数据）
        ax2.set_title("HPWL趋势（单次竞赛）")
        ax2.set_xlabel("智能体")
        ax2.set_ylabel("HPWL")
        
        # 简单趋势：每个代理的最终HPWL
        x = range(len(agent_ids))
        ax2.bar(x, hpwls, color='skyblue', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"代理{i+1}\n{name}" for i, name in enumerate(agent_names)], rotation=45, ha='right')
        
        # 突出最佳代理
        best_idx = min(range(len(hpwls)), key=lambda k: hpwls[k])
        ax2.bar(best_idx, hpwls[best_idx], color='gold', alpha=1.0, width=0.6, label="最佳")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "layout_competition_results.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"布局竞赛结果图已保存到 {results_dir}/layout_competition_results.png")

    def visualize_hpwl_improvement(self, experiment_results: Dict[int, float], baseline_hpwl: float):
        """可视化HPWL随智能体数量的改进对比图"""
        # 创建统一结果文件夹
        results_dir = "0_layout/code/layout_result"
        os.makedirs(results_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("多智能体数量对HPWL改进的影响", fontsize=14)
        
        # 提取数据
        agent_counts = sorted(experiment_results.keys())
        hpwls = [experiment_results[count] for count in agent_counts]
        improvements = [0 if count == 0 else ((baseline_hpwl - hpwls[i]) / baseline_hpwl * 100) 
                        for i, count in enumerate(agent_counts)]
        
        # 1. HPWL vs 代理数量
        colors = plt.cm.viridis(np.linspace(0, 1, len(agent_counts)))
        bars1 = ax1.bar(agent_counts, hpwls, color=colors, alpha=0.7)
        ax1.axhline(y=baseline_hpwl, color='red', linestyle='--', label='原始HPWL')
        ax1.set_xlabel("智能体数量")
        ax1.set_ylabel("HPWL")
        ax1.set_title("HPWL vs 智能体数量")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, hpwl in zip(bars1, hpwls):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{hpwl:.2f}", ha='center', va='bottom', fontsize=9)
        
        # 2. 改进百分比 vs 代理数量
        bars2 = ax2.bar(agent_counts, improvements, color='green', alpha=0.7)
        ax2.set_xlabel("智能体数量")
        ax2.set_ylabel("改进百分比 (%)")
        ax2.set_title("HPWL改进百分比")
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, imp in zip(bars2, improvements):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{imp:.1f}%", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "hpwl_improvement_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"HPWL改进对比图已保存到 {results_dir}/hpwl_improvement_comparison.png")

    def visualize_gap_analysis(self, original_modules, optimized_modules):
        """可视化缝隙分析：对比优化前后的模块间距分布"""
        # 创建结果文件夹
        results_dir = "0_layout/code/layout_result"
        os.makedirs(results_dir, exist_ok=True)

        # 计算原始和优化后的缝隙统计
        original_gaps = []
        optimized_gaps = []

        # 原始模块缝隙
        for i in range(len(original_modules)):
            for j in range(i + 1, len(original_modules)):
                gap = self.calculate_module_gap(original_modules[i], original_modules[j])
                if gap > 0:  # 只统计非重叠的缝隙
                    original_gaps.append(gap)

        # 优化后模块缝隙
        for i in range(len(optimized_modules)):
            for j in range(i + 1, len(optimized_modules)):
                gap = self.calculate_module_gap(optimized_modules[i], optimized_modules[j])
                if gap > 0:  # 只统计非重叠的缝隙
                    optimized_gaps.append(gap)

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("模块间缝隙分析对比", fontsize=14)

        # 1. 缝隙分布直方图
        if original_gaps and optimized_gaps:
            ax1.hist(original_gaps, bins=20, alpha=0.7, label='原始布局', color='lightblue', density=True)
            ax1.hist(optimized_gaps, bins=20, alpha=0.7, label='优化后布局', color='lightgreen', density=True)
            ax1.set_xlabel("模块间距 (um)")
            ax1.set_ylabel("密度")
            ax1.set_title("模块间距分布对比")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 缝隙统计对比
        if original_gaps and optimized_gaps:
            categories = ['平均间距', '最小间距', '最大间距']
            original_stats = [sum(original_gaps) / len(original_gaps), min(original_gaps), max(original_gaps)]
            optimized_stats = [sum(optimized_gaps) / len(optimized_gaps), min(optimized_gaps), max(optimized_gaps)]

            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax2.bar(x - width / 2, original_stats, width, label='原始布局', color='lightblue', alpha=0.7)
            bars2 = ax2.bar(x + width / 2, optimized_stats, width, label='优化后布局', color='lightgreen', alpha=0.7)

            ax2.set_xlabel("统计指标")
            ax2.set_ylabel("间距 (um)")
            ax2.set_title("缝隙统计对比")
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "gap_analysis_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"缝隙分析对比图已保存到 {results_dir}/gap_analysis_comparison.png")

    # =====================
    # 后处理：间距压缩框架
    # =====================
    def post_process_spacing_compression(self, modules: List[PhysicalModule]) -> List[PhysicalModule]:
        """后处理：按BFS分层从内向外压缩并校准间距，逐层可视化。
        
        在此阶段，边框（spacing guard）被废弃，只使用模块的实际尺寸。
        
        重要：此函数应该在阶段一完成后调用，使用阶段一优化后的模块作为输入。
        """
        if not modules:
            print("[阶段二] 警告：输入模块列表为空，跳过处理")
            return modules

        print("\n=== 阶段2：后处理间距压缩（废弃边框） ===")
        print("在此阶段，边框被废弃，使用模块的实际尺寸进行间距调整")
        print(f"[阶段二] 输入模块数量: {len(modules)}")
        
        # ===== 输入验证：确保使用阶段一完成后的模块 =====
        print("\n[阶段二输入验证] 验证输入模块状态...")
        
        # 验证守护边框状态（阶段一应该已激活）
        guard_active_count = sum(1 for m in modules 
                                 if getattr(m, 'spacing_guard_active', False))
        print(f"  - 守护边框激活状态: {guard_active_count}/{len(modules)} 激活")
        
        if guard_active_count == 0:
            print("  [警告] 所有模块的守护边框都未激活，可能不是阶段一的结果")
        elif guard_active_count < len(modules):
            print(f"  [警告] 有 {len(modules) - guard_active_count} 个模块的守护边框未激活")
        
        # 验证模块是否有有效位置
        invalid_positions = []
        for module in modules:
            if not hasattr(module, 'x') or not hasattr(module, 'y'):
                invalid_positions.append(module.id)
            elif not (isinstance(module.x, (int, float)) and isinstance(module.y, (int, float))):
                invalid_positions.append(module.id)
        
        if invalid_positions:
            print(f"  [错误] 发现 {len(invalid_positions)} 个模块位置无效: {invalid_positions[:5]}")
            return modules
        else:
            print("  [通过] 所有模块位置有效")
        
        # 保存输入模块的初始位置（用于后续验证）
        input_positions = {m.id: (m.x, m.y) for m in modules}
        input_hpwl = self._calculate_current_hpwl(modules)
        print(f"  - 输入模块的初始HPWL: {input_hpwl:.2f}")
        print(f"  [通过] 输入验证完成，开始阶段二处理")
        
        # 废弃所有模块的边框（这是阶段二的关键步骤）
        print("\n[阶段二] 废弃所有模块的守护边框...")
        for module in modules:
            module.spacing_guard_active = False
        print(f"  [完成] 已废弃 {len(modules)} 个模块的守护边框")
        
        # 分层：确保从内向外逐步展开
        print("\n[阶段2] 开始分层处理...")
        center_macro = self.find_center_macro(modules)
        if center_macro is None:
            print("[错误] 无法找到中心模块，跳过分层处理")
            return modules
        
        print(f"[阶段2] 中心模块: {center_macro.id} (位置: {center_macro.get_center()})")
        layer_info = self.bfs_layering(center_macro, modules)
        
        # 验证分层结果
        if len(layer_info) == 0:
            print("[错误] 分层失败，返回原模块列表")
            return modules
        
        print(f"[阶段2] 分层完成：共 {len(layer_info)} 层")
        for i, layer in enumerate(layer_info):
            print(f"  第{i}层: {len(layer)} 个模块")

        # 初始可视化
        try:
            self.visualize_layer_progress(modules, layer_info, filename="layer_progress_initial.png")
        except Exception:
            pass

        # 将列表转换为可替换的映射，便于更新
        id_to_module = {m.id: m for m in modules}

        # 从第1层（索引1）开始处理，确保从内向外逐步展开
        print("\n[阶段2] 开始逐层处理（从内向外）...")
        for layer_idx in range(1, len(layer_info)):
            # 关键修复：从id_to_module获取最新的模块引用，而不是从layer_info
            # 这样可以确保内层模块的位置是处理后的最新位置
            current_layer = [id_to_module[m.id] for m in layer_info[layer_idx]]
            inner_layers = [id_to_module[m.id] for li in layer_info[:layer_idx] for m in li]
            
            # 计算当前层和内层的统计信息
            center_pos = center_macro.get_center()
            def dist_to_center(m): 
                pos = m.get_center()
                return math.sqrt((pos[0]-center_pos[0])**2 + (pos[1]-center_pos[1])**2)
            
            inner_avg_dist = sum(dist_to_center(m) for m in inner_layers) / len(inner_layers) if inner_layers else 0
            current_avg_dist = sum(dist_to_center(m) for m in current_layer) / len(current_layer) if current_layer else 0

            print("=" * 60)
            print(f"[后处理] 开始处理第{layer_idx}层（从内向外），共 {len(current_layer)} 个模块")
            print(f"  内层模块数: {len(inner_layers)}, 平均距离中心: {inner_avg_dist:.2f}um")
            print(f"  当前层模块: {[m.id for m in current_layer]}, 平均距离中心: {current_avg_dist:.2f}um")
            
            # 验证：当前层应该比内层距离中心更远
            if inner_layers and current_avg_dist < inner_avg_dist * 0.95:
                print(f"  [警告] 当前层平均距离({current_avg_dist:.2f})小于内层({inner_avg_dist:.2f})，可能存在分层问题")

            # 获取所有模块的最新列表（用于重叠检查）
            all_modules_list = list(id_to_module.values())
            
            # 关键修复：记录内层模块的初始位置，确保在处理外层时内层模块位置不变
            inner_layer_positions = {}
            for inner in inner_layers:
                inner_mod = id_to_module.get(inner.id, inner)
                inner_layer_positions[inner_mod.id] = (inner_mod.x, inner_mod.y)
            
            # 计算当前层的初始间距统计和HPWL
            initial_gaps = []
            initial_violations = []
            initial_hpwl = self._calculate_current_hpwl(list(id_to_module.values()))
            
            # 构建连接权重映射（用于HPWL优化）
            connection_weights = self._calculate_connection_weights()
            
            for outer in current_layer:
                outer_mod = id_to_module.get(outer.id, outer)
                for inner in inner_layers:
                    inner_mod = id_to_module.get(inner.id, inner)
                    gap = self.calculate_module_gap(outer_mod, inner_mod)
                    target = self.calculate_target_spacing(outer_mod, inner_mod)
                    initial_gaps.append(gap)
                    if gap < target:
                        initial_violations.append((outer_mod.id, inner_mod.id, gap, target))
            
            if initial_gaps:
                initial_avg_gap = sum(initial_gaps) / len(initial_gaps)
                initial_min_gap = min(initial_gaps)
                print(f"  初始间距统计: 平均={initial_avg_gap:.6f}um, 最小={initial_min_gap:.6f}um, 违规数={len(initial_violations)}")
                print(f"  初始HPWL: {initial_hpwl:.2f}")
            
            # 渐进式间距调整：多轮迭代逐步缩小间距
            max_iterations = 5  # 每层最多5轮迭代
            convergence_threshold = 0.01  # 收敛阈值（um）
            
            for iteration in range(max_iterations):
                improved_count = 0
                total_improvement = 0.0
                
                # 关键修复：在每次迭代开始时，重新从id_to_module获取最新的模块引用
                # 这样可以确保内层模块的位置是处理后的最新位置（虽然内层不应该被移动）
                current_layer = [id_to_module[m.id] for m in layer_info[layer_idx]]
                inner_layers = [id_to_module[m.id] for li in layer_info[:layer_idx] for m in li]
                
                # 逐模块与内层邻居校准目标间距（从内到外严格推进）
                # 注意：current_layer 和 inner_layers 在每次迭代都从 id_to_module 获取最新引用
                for outer in current_layer:
                    # 确保使用最新的模块实例
                    outer_mod = id_to_module.get(outer.id, outer)

                    # 查找内层邻居：与内层任一模块几何相邻
                    # 在后处理阶段（边框废弃后），使用目标间距的2.0倍作为邻居检测阈值
                    # 注意：inner_layers 在每次迭代都从 id_to_module 获取最新引用
                    neighbors = []
                    for inner in inner_layers:
                        # 确保使用最新的模块实例（双重保险）
                        inner_mod = id_to_module.get(inner.id, inner)
                        gap = self.calculate_module_gap(outer_mod, inner_mod)
                        target_spacing = self.calculate_target_spacing(outer_mod, inner_mod)
                        
                        # 动态阈值：目标间距的2.0倍（更宽松，确保找到所有需要调整的）
                        neighbor_threshold = target_spacing * 2.0
                        
                        if gap <= neighbor_threshold:
                            neighbors.append(inner_mod)

                    # 按连接权重排序邻居（优先处理有连接的模块对，以降低HPWL）
                    neighbors_with_priority = []
                    for nb in neighbors:
                        # 检查是否有连接
                        has_connection = False
                        connection_weight = 0.0
                        
                        # 检查连接权重
                        key1 = (outer_mod.id, nb.id)
                        key2 = (nb.id, outer_mod.id)
                        if key1 in connection_weights:
                            connection_weight = connection_weights[key1]
                            has_connection = True
                        elif key2 in connection_weights:
                            connection_weight = connection_weights[key2]
                            has_connection = True
                        
                        # 优先级：有连接的模块对优先，权重高的更优先
                        priority = connection_weight if has_connection else 0.0
                        neighbors_with_priority.append((nb, priority, has_connection))
                    
                    # 按优先级排序：有连接的优先，权重高的优先
                    neighbors_with_priority.sort(key=lambda x: (x[2], x[1]), reverse=True)
                    
                    # 顺序与每个内层邻居调整到目标间距（严格禁止重叠，优先降低HPWL）
                    for nb, priority, has_conn in neighbors_with_priority:
                        before_gap = self.calculate_module_gap(outer_mod, nb)
                        target = self.calculate_target_spacing(outer_mod, nb)

                        # 计算间距差异
                        gap_diff = abs(before_gap - target)
                        
                        # 渐进式调整：每轮只调整一部分差距，逐步缩小
                        # 调整因子随迭代次数递减：第1轮调整80%，第2轮调整60%，以此类推
                        # 对于有连接的模块对，使用更大的调整因子（更快缩小间距以降低HPWL）
                        base_adjustment_factor = max(0.3, 0.8 - iteration * 0.15)  # 0.8, 0.65, 0.5, 0.35, 0.3
                        if has_conn:
                            # 有连接的模块对：增加调整因子，更快缩小间距
                            adjustment_factor = min(0.95, base_adjustment_factor * 1.2)
                        else:
                            adjustment_factor = base_adjustment_factor
                        
                        # 只有当间距差异显著时才调整
                        if gap_diff > convergence_threshold:
                            # 计算本轮目标间距（渐进式）
                            if before_gap < target:
                                # 间距太小，需要推远（但只推到目标值，不推更远）
                                partial_target = before_gap + (target - before_gap) * adjustment_factor
                                moved = self.move_module_away(outer_mod, nb, partial_target, all_modules=all_modules_list)
                                action = "推远"
                            else:
                                # 间距太大，需要拉近（缩小间距以降低HPWL）
                                # 对于有连接的模块对，可以尝试缩小到略小于目标值（在规则允许范围内）
                                if has_conn and before_gap > target * 1.1:
                                    # 有连接且间距较大：尝试缩小到目标值的95%（在规则允许范围内）
                                    optimal_target = target * 0.95  # 略小于目标值，但仍在规则范围内
                                    partial_target = before_gap - (before_gap - optimal_target) * adjustment_factor
                                else:
                                    # 无连接或间距接近目标：缩小到目标值
                                    partial_target = before_gap - (before_gap - target) * adjustment_factor
                                
                                moved = self.move_module_closer(outer_mod, nb, partial_target, all_modules=all_modules_list)
                                action = "拉近"
                            
                            # 严格检查：移动后是否与任何模块重叠（双重保险）
                            has_overlap, overlapping_ids = self.check_overlap_with_all_modules(
                                moved, all_modules_list, exclude_ids={moved.id}
                            )
                            
                            if has_overlap:
                                # 如果移动后产生重叠，拒绝移动
                                continue
                            
                            # 检查移动后与同层其他模块的间距，确保不会严重恶化
                            should_move = True
                            for other in current_layer:
                                if other.id == outer.id:
                                    continue
                                other_latest = id_to_module.get(other.id, other)
                                
                                # 计算移动前后与同层模块的间距
                                gap_before = self.calculate_module_gap(outer_mod, other_latest)
                                gap_after = self.calculate_module_gap(moved, other_latest)
                                
                                # 如果移动后与同层模块重叠，拒绝移动
                                overlap_with_other, _ = self.check_module_overlap(moved, other_latest)
                                if overlap_with_other:
                                    should_move = False
                                    break
                                
                                # 如果移动导致间距显著恶化，也拒绝移动
                                inner_gap_improvement = abs(target - before_gap)
                                deterioration_threshold = min(20.0, inner_gap_improvement * 0.2)  # 更宽松的阈值
                                if gap_after < gap_before - deterioration_threshold:
                                    should_move = False
                                    break
                            
                            if should_move:
                                after_gap = self.calculate_module_gap(moved, nb)
                                improvement = abs(after_gap - target) - abs(before_gap - target)
                                
                                # 计算HPWL变化（如果模块有连接）
                                hpwl_before = 0.0
                                hpwl_after = 0.0
                                if has_conn:
                                    # 临时更新模块位置计算HPWL
                                    old_pos = (outer_mod.x, outer_mod.y)
                                    id_to_module[moved.id] = moved
                                    temp_modules = list(id_to_module.values())
                                    hpwl_after = self._calculate_current_hpwl(temp_modules)
                                    
                                    # 恢复原位置计算HPWL
                                    id_to_module[outer_mod.id] = outer_mod
                                    hpwl_before = self._calculate_current_hpwl(list(id_to_module.values()))
                                    
                                    # 恢复新位置
                                    id_to_module[moved.id] = moved
                                
                                # 更新映射与层引用对象
                                id_to_module[moved.id] = moved
                                outer_mod = moved
                                # 更新all_modules_list中的引用
                                all_modules_list = list(id_to_module.values())
                                
                                if improvement < 0:  # 间距更接近目标
                                    improved_count += 1
                                    total_improvement += abs(improvement)
                                    
                                    # 如果有连接且HPWL降低，额外记录
                                    if has_conn and hpwl_after < hpwl_before:
                                        hpwl_reduction = hpwl_before - hpwl_after
                                        print(f"      -> [HPWL优化] {action} {outer_mod.id} 降低HPWL {hpwl_reduction:.2f}")
                
                # 计算本轮迭代后的间距统计和HPWL
                current_gaps = []
                current_violations = []
                for outer in current_layer:
                    outer_mod = id_to_module.get(outer.id, outer)
                    for inner in inner_layers:
                        inner_mod = id_to_module.get(inner.id, inner)
                        gap = self.calculate_module_gap(outer_mod, inner_mod)
                        target = self.calculate_target_spacing(outer_mod, inner_mod)
                        current_gaps.append(gap)
                        if gap < target:
                            current_violations.append((outer_mod.id, inner_mod.id, gap, target))
                
                current_hpwl = self._calculate_current_hpwl(list(id_to_module.values()))
                hpwl_improvement = initial_hpwl - current_hpwl
                
                if current_gaps:
                    current_avg_gap = sum(current_gaps) / len(current_gaps)
                    current_min_gap = min(current_gaps)
                    violation_reduction = len(initial_violations) - len(current_violations)
                    
                    print(f"  迭代 {iteration+1}/{max_iterations}: 改进{improved_count}个间距, "
                          f"平均间距={current_avg_gap:.6f}um, 最小间距={current_min_gap:.6f}um, "
                          f"违规减少={violation_reduction}, HPWL={current_hpwl:.2f} "
                          f"({hpwl_improvement:+.2f})")
                
                # 如果本轮没有改进或已收敛，提前结束
                if improved_count == 0:
                    print(f"  迭代 {iteration+1}: 无进一步改进，提前结束")
                    break
                
                # 检查是否已收敛（所有间距都接近目标）
                max_gap_diff = 0.0
                for outer in current_layer:
                    outer_mod = id_to_module.get(outer.id, outer)
                    for inner in inner_layers:
                        inner_mod = id_to_module.get(inner.id, inner)
                        gap = self.calculate_module_gap(outer_mod, inner_mod)
                        target = self.calculate_target_spacing(outer_mod, inner_mod)
                        max_gap_diff = max(max_gap_diff, abs(gap - target))
                
                if max_gap_diff < convergence_threshold:
                    print(f"  迭代 {iteration+1}: 已收敛（最大间距差异={max_gap_diff:.6f}um < {convergence_threshold}um）")
                    break
            
            # 最终间距统计和HPWL
            final_gaps = []
            final_violations = []
            for outer in current_layer:
                outer_mod = id_to_module.get(outer.id, outer)
                for inner in inner_layers:
                    inner_mod = id_to_module.get(inner.id, inner)
                    gap = self.calculate_module_gap(outer_mod, inner_mod)
                    target = self.calculate_target_spacing(outer_mod, inner_mod)
                    final_gaps.append(gap)
                    if gap < target:
                        final_violations.append((outer_mod.id, inner_mod.id, gap, target))
            
            final_hpwl = self._calculate_current_hpwl(list(id_to_module.values()))
            total_hpwl_improvement = initial_hpwl - final_hpwl
            
            if final_gaps:
                final_avg_gap = sum(final_gaps) / len(final_gaps)
                final_min_gap = min(final_gaps)
                gap_reduction = initial_avg_gap - final_avg_gap if initial_gaps else 0
                violation_reduction = len(initial_violations) - len(final_violations)
                
                print(f"  最终间距统计: 平均={final_avg_gap:.6f}um, 最小={final_min_gap:.6f}um, "
                      f"平均间距减少={gap_reduction:.6f}um, 违规减少={violation_reduction}")
                print(f"  最终HPWL: {final_hpwl:.2f} (改进 {total_hpwl_improvement:+.2f}, "
                      f"{total_hpwl_improvement/initial_hpwl*100:+.2f}%)")
                
                if final_violations:
                    print(f"  [警告] 仍有 {len(final_violations)} 个间距违规未解决")
            
            # 新增：HPWL优化阶段（在满足规则的前提下进一步缩小间距）
            print(f"\n  [HPWL优化] 开始第{layer_idx}层的HPWL优化...")
            optimized_modules, hpwl_after_opt = self._optimize_layer_hpwl(
                current_layer, inner_layers, id_to_module, all_modules_list, connection_weights
            )
            
            # 更新模块引用
            for mod in optimized_modules:
                id_to_module[mod.id] = mod
            all_modules_list = list(id_to_module.values())
            
            if hpwl_after_opt < final_hpwl:
                hpwl_opt_improvement = final_hpwl - hpwl_after_opt
                print(f"  [HPWL优化] 完成: HPWL从{final_hpwl:.2f}降至{hpwl_after_opt:.2f} "
                      f"(额外改进{hpwl_opt_improvement:.2f}, {hpwl_opt_improvement/final_hpwl*100:.2f}%)")
            else:
                print(f"  [HPWL优化] 完成: HPWL={hpwl_after_opt:.2f} (无进一步改进)")

            # 层处理完成后，用最新对象刷新本层与后续层的实例引用
            # 关键：只更新当前层及后续层的引用，内层引用保持不变（内层已经处理完成，位置应该固定）
            for li in range(layer_idx, len(layer_info)):
                layer_info[li] = [id_to_module[m.id] for m in layer_info[li]]
            
            # 验证当前层的间距是否符合规则（使用最新的模块引用）
            # 注意：这里需要重新获取最新的引用，因为current_layer和inner_layers可能在迭代中被修改
            current_layer_final = [id_to_module[m.id] for m in layer_info[layer_idx]]
            inner_layers_final = [id_to_module[m.id] for li in layer_info[:layer_idx] for m in li]
            layer_compliance = self._verify_layer_spacing_compliance(
                current_layer_final, inner_layers_final, id_to_module
            )
            print(f"  第{layer_idx}层间距合规性: {layer_compliance['compliance_rate']:.1f}% "
                  f"({layer_compliance['compliant_pairs']}/{layer_compliance['total_pairs']} 对符合规则)")
            
            # 验证内层模块位置是否被意外改变（关键检查）
            if layer_idx > 1 and inner_layer_positions:
                # 检查内层模块的位置是否在处理当前层时被改变
                # 这不应该发生，因为只有外层模块被移动
                position_changed_count = 0
                for inner_id, (orig_x, orig_y) in inner_layer_positions.items():
                    current_mod = id_to_module.get(inner_id)
                    if current_mod:
                        current_x, current_y = current_mod.x, current_mod.y
                        # 允许微小的浮点数误差（1e-6）
                        if abs(current_x - orig_x) > 1e-6 or abs(current_y - orig_y) > 1e-6:
                            position_changed_count += 1
                            print(f"  [警告] 内层模块 {inner_id} 位置被改变: "
                                  f"({orig_x:.6f}, {orig_y:.6f}) -> ({current_x:.6f}, {current_y:.6f})")
                
                if position_changed_count == 0:
                    print(f"  [验证] 内层模块位置未改变（符合预期）")
                else:
                    print(f"  [错误] 发现 {position_changed_count} 个内层模块位置被意外改变！")

            # 可视化该层处理后的进度
            try:
                filename = f"layer_{layer_idx}_adjustment.png"
                # 使用最新的模块列表和层信息进行可视化
                self.visualize_layer_progress(list(id_to_module.values()), layer_info, filename=filename)
            except Exception as e:
                print(f"[可视化警告] 生成第{layer_idx}层进度图失败: {e}")
        
        # 最终验证：所有层的间距合规性
        print("\n" + "=" * 60)
        print("[阶段2] 最终间距合规性验证")
        print("=" * 60)
        total_compliant = 0
        total_pairs = 0
        for layer_idx in range(1, len(layer_info)):
            # 使用最新的模块引用进行验证
            current_layer = [id_to_module[m.id] for m in layer_info[layer_idx]]
            inner_layers = [id_to_module[m.id] for li in layer_info[:layer_idx] for m in li]
            compliance = self._verify_layer_spacing_compliance(
                current_layer, inner_layers, id_to_module
            )
            total_compliant += compliance['compliant_pairs']
            total_pairs += compliance['total_pairs']
            print(f"第{layer_idx}层: {compliance['compliance_rate']:.1f}% "
                  f"({compliance['compliant_pairs']}/{compliance['total_pairs']} 对)")
        
        overall_compliance = (total_compliant / total_pairs * 100) if total_pairs > 0 else 0
        print(f"\n总体合规性: {overall_compliance:.1f}% ({total_compliant}/{total_pairs} 对符合规则)")

        # ===== 阶段二完成验证：确保基于阶段一的结果进行处理 =====
        print("\n" + "=" * 60)
        print("[阶段二完成验证] 验证处理结果")
        print("=" * 60)
        
        # 返回最新的模块列表（保持输入顺序）
        final_modules = [id_to_module[m.id] for m in modules]
        
        # 验证最终模块数量
        if len(final_modules) != len(modules):
            print(f"[错误] 模块数量发生变化: {len(modules)} -> {len(final_modules)}")
        else:
            print(f"[通过] 模块数量保持一致: {len(final_modules)}")
        
        # 验证守护边框状态（应该全部废弃）
        final_guard_active = sum(1 for m in final_modules 
                                 if getattr(m, 'spacing_guard_active', False))
        if final_guard_active == 0:
            print(f"[通过] 所有模块的守护边框已废弃（符合阶段二预期）")
        else:
            print(f"[警告] 仍有 {final_guard_active} 个模块的守护边框未废弃")
        
        # 计算最终HPWL
        final_hpwl = self._calculate_current_hpwl(final_modules)
        hpwl_change = final_hpwl - input_hpwl
        print(f"[结果] 最终HPWL: {final_hpwl:.2f} (相对于输入: {hpwl_change:+.2f})")
        
        # 验证位置变化（与输入位置对比）
        position_changes = []
        for module in final_modules:
            if module.id in input_positions:
                orig_x, orig_y = input_positions[module.id]
                if abs(module.x - orig_x) > 1e-6 or abs(module.y - orig_y) > 1e-6:
                    dx, dy = module.x - orig_x, module.y - orig_y
                    position_changes.append((module.id, dx, dy))
        
        if position_changes:
            print(f"[结果] 位置变化: {len(position_changes)}/{len(final_modules)} 个模块位置发生变化")
            print(f"  [说明] 这是正常的，阶段二会调整模块位置以满足DRC规则")
            # 显示变化最大的5个模块
            position_changes.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)
            for mod_id, dx, dy in position_changes[:5]:
                print(f"    模块 {mod_id}: Δx={dx:.6f}, Δy={dy:.6f}")
        else:
            print(f"[结果] 所有模块位置保持不变（可能未进行实际调整）")
        
        print("=" * 60)
        print("\n[后处理] 分层间距压缩完成。")
        return final_modules
    
    def _verify_layer_spacing_compliance(self, current_layer: List[PhysicalModule], 
                                        inner_layers: List[PhysicalModule],
                                        id_to_module: Dict[str, PhysicalModule]) -> Dict[str, Any]:
        """验证当前层与内层之间的间距是否符合DRC规则
        
        返回:
            包含合规性统计的字典
        """
        compliant_pairs = 0
        total_pairs = 0
        violations = []
        
        for outer in current_layer:
            outer_mod = id_to_module.get(outer.id, outer)
            for inner in inner_layers:
                inner_mod = id_to_module.get(inner.id, inner)
                
                # 计算实际间距和目标间距
                actual_gap = self.calculate_module_gap(outer_mod, inner_mod)
                target_spacing = self.calculate_target_spacing(outer_mod, inner_mod)
                
                total_pairs += 1
                
                # 检查是否符合规则（允许1%的容差）
                if actual_gap >= target_spacing * 0.99:
                    compliant_pairs += 1
                else:
                    violations.append({
                        'outer': outer_mod.id,
                        'inner': inner_mod.id,
                        'actual': actual_gap,
                        'target': target_spacing,
                        'violation': target_spacing - actual_gap
                    })
        
        compliance_rate = (compliant_pairs / total_pairs * 100) if total_pairs > 0 else 0
        
        return {
            'compliant_pairs': compliant_pairs,
            'total_pairs': total_pairs,
            'compliance_rate': compliance_rate,
            'violations': violations
        }
    
    def _optimize_layer_hpwl(self, current_layer: List[PhysicalModule], 
                             inner_layers: List[PhysicalModule],
                             id_to_module: Dict[str, PhysicalModule],
                             all_modules_list: List[PhysicalModule],
                             connection_weights: Dict[Tuple[str, str], float]) -> Tuple[List[PhysicalModule], float]:
        """在满足规则的前提下优化当前层的HPWL
        
        策略：
        1. 优先处理有连接的模块对
        2. 在满足DRC规则的前提下，尽量缩小间距到目标值的95%（在规则允许范围内）
        3. 计算HPWL变化，选择能降低HPWL的移动
        4. 确保不违反DRC规则
        
        返回:
            (优化后的模块列表, 优化后的HPWL)
        """
        # 使用传入的模块列表（已经是处理后的最新状态）
        optimized_modules = [id_to_module[m.id] for m in current_layer]
        module_map = {m.id: m for m in optimized_modules}
        
        initial_hpwl = self._calculate_current_hpwl(all_modules_list)
        best_hpwl = initial_hpwl
        best_modules = [m.copy() for m in optimized_modules]
        
        # 收集有连接的模块对（优先处理）
        connected_pairs = []
        for outer in optimized_modules:
            for inner in inner_layers:
                inner_mod = id_to_module.get(inner.id, inner)
                
                # 检查是否有连接
                key1 = (outer.id, inner_mod.id)
                key2 = (inner_mod.id, outer.id)
                weight = 0.0
                if key1 in connection_weights:
                    weight = connection_weights[key1]
                elif key2 in connection_weights:
                    weight = connection_weights[key2]
                
                if weight > 0:
                    gap = self.calculate_module_gap(outer, inner_mod)
                    target = self.calculate_target_spacing(outer, inner_mod)
                    # 只处理间距大于目标值的情况（可以缩小间距）
                    if gap > target * 1.05:
                        connected_pairs.append((outer, inner_mod, weight, gap, target))
        
        # 按权重排序（权重高的优先）
        connected_pairs.sort(key=lambda x: x[2], reverse=True)
        
        if not connected_pairs:
            print(f"    未找到可优化的连接对（所有连接对间距已接近目标值）")
            return best_modules, best_hpwl
        
        print(f"    找到 {len(connected_pairs)} 个有连接的模块对，开始优化...")
        
        # 对每个有连接的模块对进行优化
        improvements = 0
        total_hpwl_reduction = 0.0
        
        for outer, inner, weight, gap, target in connected_pairs:
            outer_mod = module_map.get(outer.id, outer)
            
            # 计算优化目标间距（目标值的95%，确保在规则范围内）
            optimal_target = target * 0.95
            
            # 尝试拉近模块
            moved = self.move_module_closer(outer_mod, inner, optimal_target, all_modules=all_modules_list)
            
            # 验证移动后是否满足规则
            after_gap = self.calculate_module_gap(moved, inner)
            if after_gap >= target * 0.99:  # 确保仍在规则范围内（99%容差）
                # 检查重叠
                has_overlap, _ = self.check_overlap_with_all_modules(
                    moved, all_modules_list, exclude_ids={moved.id}
                )
                
                if not has_overlap:
                    # 计算HPWL变化
                    old_hpwl = self._calculate_current_hpwl(all_modules_list)
                    
                    # 临时更新模块位置
                    module_map[moved.id] = moved
                    optimized_modules = [module_map.get(m.id, m) for m in optimized_modules]
                    
                    # 更新all_modules_list以计算新HPWL
                    temp_all = []
                    for m in all_modules_list:
                        if m.id == moved.id:
                            temp_all.append(moved)
                        elif m.id in module_map:
                            temp_all.append(module_map[m.id])
                        else:
                            temp_all.append(m)
                    
                    new_hpwl = self._calculate_current_hpwl(temp_all)
                    
                    if new_hpwl < old_hpwl:
                        # HPWL降低，接受移动
                        best_hpwl = new_hpwl
                        best_modules = [m.copy() for m in optimized_modules]
                        improvements += 1
                        hpwl_reduction = old_hpwl - new_hpwl
                        total_hpwl_reduction += hpwl_reduction
                        
                        # 更新id_to_module和all_modules_list
                        id_to_module[moved.id] = moved
                        all_modules_list = temp_all
                        
                        print(f"      -> [HPWL优化] {outer_mod.id}-{inner.id}: "
                              f"间距{gap:.2f}->{after_gap:.2f}um, HPWL降低{hpwl_reduction:.2f}")
                    else:
                        # HPWL未降低，恢复
                        module_map[outer_mod.id] = outer_mod
                        optimized_modules = [module_map.get(m.id, m) for m in optimized_modules]
                else:
                    # 有重叠，恢复
                    module_map[outer_mod.id] = outer_mod
            else:
                # 移动后不满足规则，恢复
                module_map[outer_mod.id] = outer_mod
        
        if improvements > 0:
            print(f"    HPWL优化完成: 改进了{improvements}个连接对, 总HPWL降低{total_hpwl_reduction:.2f}")
        else:
            print(f"    未找到可优化的连接对（所有移动都会导致HPWL增加或违反规则）")
        
        return best_modules, best_hpwl

    def find_center_macro(self, modules: List[PhysicalModule]) -> PhysicalModule:
        """找到几何中心附近的模块（改进版：确保找到真正的中心模块）。

        规则：
        1. 计算所有可移动模块的几何质心
        2. 计算每个模块到质心的距离
        3. 选择距离质心最近的模块作为中心模块
        4. 如果全部为固定模块，选择面积最大的模块
        """
        if not modules:
            return None
        
        movable = [m for m in modules if not getattr(m, 'is_fixed', False) and m.type != 'obstacle']
        candidates = movable if movable else modules
        
        if not candidates:
            # 如果全部为固定模块，选择面积最大的
            return max(modules, key=lambda m: m.width * m.height)

        # 计算所有候选模块的几何质心
        centers = [m.get_center() for m in candidates]
        cx = sum(p[0] for p in centers) / len(centers)
        cy = sum(p[1] for p in centers) / len(centers)
        
        # 计算每个模块到质心的欧氏距离
        def distance_to_center(module: PhysicalModule) -> float:
            center = module.get_center()
            return math.sqrt((center[0] - cx)**2 + (center[1] - cy)**2)
        
        # 选择距离质心最近的模块
        best = min(candidates, key=distance_to_center)
        
        print(f"[中心模块] 选择模块 {best.id} 作为中心（距离质心 {distance_to_center(best):.2f}um）")
        return best

    def bfs_layering(self, center_macro: PhysicalModule, modules: List[PhysicalModule]) -> List[List[PhysicalModule]]:
        """使用改进的BFS分层：严格基于距离中心模块的层次进行分层，确保从内向外逐步展开。

        分层策略（改进版）：
        1. 第0层：中心模块
        2. 第1层：与中心模块相邻的模块（gap <= min_spacing），按距离排序
        3. 第2层：与第1层相邻但未访问的模块，按距离排序
        4. 如果BFS无法保证距离递增，使用纯距离分层
        5. 后处理：重新分配模块到层，确保每层距离严格递增
        
        关键改进：
        - 每层生成时，严格验证距离是否递增
        - 如果距离不递增，使用距离分层重新分配
        - 最终后处理，确保所有层严格从内到外
        """
        if not modules:
            return []
        if center_macro not in modules:
            return [modules]

        # 构建索引方便查找
        id_to_module = {m.id: m for m in modules}
        center_pos = center_macro.get_center()

        visited: set = set()
        layers: List[List[PhysicalModule]] = []
        
        # 计算模块到中心的距离（用于验证和排序）
        def distance_to_center(module: PhysicalModule) -> float:
            pos = module.get_center()
            return math.sqrt((pos[0] - center_pos[0])**2 + (pos[1] - center_pos[1])**2)

        # BFS初始化：第0层是中心模块
        current_layer_ids = [center_macro.id]
        visited.add(center_macro.id)

        layer_idx = 0
        max_layers = 10  # 最大层数限制，防止无限循环
        
        while current_layer_ids and layer_idx < max_layers:
            # 生成当前层模块对象
            current_layer_modules = [id_to_module[mid] for mid in current_layer_ids]
            
            # 计算当前层到中心的距离统计
            current_distances = [distance_to_center(m) for m in current_layer_modules]
            avg_distance = sum(current_distances) / len(current_distances)
            min_distance = min(current_distances)
            max_distance = max(current_distances)

            # 验证：当前层的距离应该比前一层更远（除了第0层）
            if layer_idx > 0 and len(layers) > 0:
                prev_layer = layers[-1]
                prev_distances = [distance_to_center(m) for m in prev_layer]
                prev_avg = sum(prev_distances) / len(prev_distances)
                prev_min = min(prev_distances)
                
                # 严格检查：当前层的最小距离必须大于前一层的最小距离
                if min_distance <= prev_min * 0.95:  # 允许5%容差
                    print(f"  [警告] 第{layer_idx}层最小距离({min_distance:.2f}) <= 第{layer_idx-1}层最小距离({prev_min:.2f})")
                    print(f"  [修复] 使用距离分层重新分配...")
                    
                    # 使用距离分层重新分配：将所有未访问模块按距离分组
                    remaining = [m for m in modules if m.id not in visited]
                    if remaining:
                        remaining_with_dist = [(m, distance_to_center(m)) for m in remaining]
                        remaining_with_dist.sort(key=lambda x: x[1])
                        
                        # 找到距离大于前一层最小距离的模块
                        distance_threshold = prev_min * 1.05  # 至少比前一层远5%
                        valid_next = [(m, d) for m, d in remaining_with_dist if d >= distance_threshold]
                        
                        if valid_next:
                            # 将距离相近的模块归为一层（容差15%）
                            next_min_dist = valid_next[0][1]
                            threshold = next_min_dist * 1.15
                            
                            for m, dist in valid_next:
                                if dist <= threshold:
                                    visited.add(m.id)
                                    current_layer_ids.append(m.id)
                                else:
                                    break
                            
                            # 重新生成当前层
                            current_layer_modules = [id_to_module[mid] for mid in current_layer_ids]
                            current_distances = [distance_to_center(m) for m in current_layer_modules]
                            avg_distance = sum(current_distances) / len(current_distances)
                            min_distance = min(current_distances)
                            max_distance = max(current_distances)
                            print(f"  [修复完成] 第{layer_idx}层重新分配: {len(current_layer_modules)}个模块, "
                                  f"最小距离={min_distance:.2f}um")
            
            # 按距离排序当前层模块（确保层内也按距离排序）
            current_layer_modules.sort(key=distance_to_center)
            layers.append(current_layer_modules)
            
            # 打印当前层信息
            print(f"[BFS分层] 第{layer_idx}层: {[m.id for m in current_layer_modules]} "
                  f"(平均距离中心: {avg_distance:.2f}um, 范围: {min_distance:.2f}-{max_distance:.2f}um)")

            # 准备下一层
            next_layer_ids: List[str] = []
            
            # 收集所有未访问的模块，按距离中心排序
            unvisited_modules = [(m, distance_to_center(m)) for m in modules 
                                if m.id not in visited]
            unvisited_modules.sort(key=lambda x: x[1])  # 按距离排序

            # 为每个当前层模块寻找未访问且相邻的模块
            for mid in current_layer_ids:
                m = id_to_module[mid]
                for other, dist in unvisited_modules:
                    if other.id in visited:
                        continue
                    if other.id == mid:
                        continue
                    
                    # 相邻判定：间隙 <= 最小间距（包含重叠）
                    gap = self.calculate_module_gap(m, other)
                    if gap <= self.min_spacing:
                        visited.add(other.id)
                        next_layer_ids.append(other.id)

            # 去重并按距离排序
            seen = set()
            unique_next_layer_ids = []
            next_layer_with_dist = [(id_to_module[oid], distance_to_center(id_to_module[oid])) 
                                   for oid in next_layer_ids if oid not in seen]
            next_layer_with_dist.sort(key=lambda x: x[1])  # 按距离排序
            
            for other, dist in next_layer_with_dist:
                if other.id not in seen:
                    seen.add(other.id)
                    unique_next_layer_ids.append(other.id)

            # 如果下一层为空，但还有未访问模块，使用距离分层
            if not unique_next_layer_ids and len(visited) < len(modules):
                remaining = [m for m in modules if m.id not in visited]
                if remaining:
                    remaining_with_dist = [(m, distance_to_center(m)) for m in remaining]
                    remaining_with_dist.sort(key=lambda x: x[1])
                    
                    # 确保下一层距离大于当前层
                    current_max_dist = max(distance_to_center(m) for m in current_layer_modules)
                    distance_threshold = current_max_dist * 1.05  # 至少比当前层最远模块远5%
                    
                    valid_next = [(m, d) for m, d in remaining_with_dist if d >= distance_threshold]
                    if valid_next:
                        next_min_dist = valid_next[0][1]
                        threshold = next_min_dist * 1.15  # 距离相近的归为一层
                        
                        for m, dist in valid_next:
                            if dist <= threshold:
                                visited.add(m.id)
                                unique_next_layer_ids.append(m.id)
                            else:
                                break
                        
                        if unique_next_layer_ids:
                            print(f"  [距离分层] 添加 {len(unique_next_layer_ids)} 个模块到第{layer_idx+1}层")

            current_layer_ids = unique_next_layer_ids
            layer_idx += 1

        # 后处理：重新验证和调整层，确保严格从内到外
        print("\n[分层后处理] 重新验证和调整层顺序...")
        layers = self._refine_layers_by_distance(layers, center_pos, id_to_module)

        # 最终验证：确保每层距离严格递增
        print("\n[分层验证] 最终验证从内到外的层次关系：")
        for i, layer in enumerate(layers):
            if layer:
                distances = [distance_to_center(m) for m in layer]
                avg_dist = sum(distances) / len(distances)
                min_dist = min(distances)
                max_dist = max(distances)
                print(f"  第{i}层: {len(layer)}个模块, "
                      f"平均距离={avg_dist:.2f}um, 最小={min_dist:.2f}um, 最大={max_dist:.2f}um")
        
        # 检查是否所有模块都被分层
        all_layered_ids = {m.id for layer in layers for m in layer}
        if len(all_layered_ids) < len(modules):
            missing = [m.id for m in modules if m.id not in all_layered_ids]
            print(f"  [警告] 有 {len(missing)} 个模块未被分层: {missing}")
            # 将未分层的模块按距离添加到合适的层
            for m in modules:
                if m.id not in all_layered_ids:
                    dist = distance_to_center(m)
                    # 找到合适的层（距离大于前一层，小于后一层）
                    added = False
                    for i in range(len(layers)):
                        if i == 0:
                            continue
                        prev_max = max(distance_to_center(m2) for m2 in layers[i-1]) if layers[i-1] else 0
                        if dist >= prev_max * 0.95:
                            if i < len(layers):
                                layers[i].append(m)
                            else:
                                layers.append([m])
                            added = True
                            break
                    if not added:
                        layers[-1].append(m)

        return layers
    
    def _refine_layers_by_distance(self, layers: List[List[PhysicalModule]], 
                                   center_pos: Tuple[float, float],
                                   id_to_module: Dict[str, PhysicalModule]) -> List[List[PhysicalModule]]:
        """后处理：根据距离重新调整层，确保严格从内到外
        
        策略：
        1. 收集所有模块及其到中心的距离
        2. 按距离排序所有模块
        3. 根据距离阈值重新分组，确保每层距离严格递增
        4. 验证新分层的距离递增关系
        """
        def distance_to_center(module: PhysicalModule) -> float:
            pos = module.get_center()
            return math.sqrt((pos[0] - center_pos[0])**2 + (pos[1] - center_pos[1])**2)
        
        if len(layers) <= 1:
            return layers
        
        # 收集所有模块及其距离
        all_modules_with_dist = []
        for layer in layers:
            for m in layer:
                all_modules_with_dist.append((m, distance_to_center(m)))
        
        # 按距离排序
        all_modules_with_dist.sort(key=lambda x: x[1])
        
        if not all_modules_with_dist:
            return layers
        
        # 重新分层：使用距离阈值确保每层距离递增
        refined_layers = []
        current_layer = []
        prev_layer_max_dist = -1.0
        
        # 第0层：中心模块（距离为0或最小）
        first_module, first_dist = all_modules_with_dist[0]
        if first_dist < 1e-6:  # 中心模块
            current_layer.append(first_module)
            prev_layer_max_dist = first_dist
            start_idx = 1
        else:
            start_idx = 0
        
        # 从第1个模块开始（或从第0个开始，如果没有中心模块）
        for i in range(start_idx, len(all_modules_with_dist)):
            m, dist = all_modules_with_dist[i]
            
            # 如果当前层为空，直接添加
            if not current_layer:
                current_layer.append(m)
                prev_layer_max_dist = dist
            else:
                # 计算当前层的距离范围
                current_distances = [distance_to_center(m2) for m2 in current_layer]
                current_min = min(current_distances)
                current_max = max(current_distances)
                current_avg = sum(current_distances) / len(current_distances)
                
                # 判断是否应该开始新层
                # 条件1：新模块距离必须大于前一层最大距离
                # 条件2：新模块距离大于当前层最小距离的1.15倍（确保层间有足够间隔）
                should_start_new_layer = False
                
                if dist > prev_layer_max_dist * 1.05:  # 至少比前一层远5%
                    # 检查是否应该归入当前层还是开始新层
                    if dist > current_min * 1.15:  # 距离差距超过15%，开始新层
                        should_start_new_layer = True
                    elif dist <= current_max * 1.1:  # 距离在容差范围内，归入当前层
                        should_start_new_layer = False
                    else:
                        # 距离在中间，根据当前层大小决定
                        if len(current_layer) >= 3:  # 当前层已有足够模块，开始新层
                            should_start_new_layer = True
                        else:
                            should_start_new_layer = False
                else:
                    # 距离太近，归入当前层（但这种情况不应该发生，因为已经排序）
                    should_start_new_layer = False
                
                if should_start_new_layer:
                    # 保存当前层
                    refined_layers.append(current_layer)
                    # 开始新层
                    current_layer = [m]
                    prev_layer_max_dist = dist
                else:
                    # 添加到当前层
                    current_layer.append(m)
                    prev_layer_max_dist = max(prev_layer_max_dist, dist)
        
        # 添加最后一层
        if current_layer:
            refined_layers.append(current_layer)
        
        # 验证新分层的距离递增
        print(f"  [后处理] 重新分层: {len(layers)}层 -> {len(refined_layers)}层")
        prev_min = -1.0
        for i, layer in enumerate(refined_layers):
            if layer:
                distances = [distance_to_center(m) for m in layer]
                avg_dist = sum(distances) / len(distances)
                min_dist = min(distances)
                max_dist = max(distances)
                
                # 验证距离递增
                if i > 0 and prev_min >= 0:
                    if min_dist <= prev_min * 0.95:
                        print(f"    [警告] 第{i}层最小距离({min_dist:.2f}) <= 第{i-1}层最小距离({prev_min:.2f})")
                    else:
                        print(f"    [验证] 第{i}层最小距离({min_dist:.2f}) > 第{i-1}层最小距离({prev_min:.2f}) ✓")
                
                print(f"    第{i}层: {len(layer)}个模块, "
                      f"平均距离={avg_dist:.2f}um, 最小={min_dist:.2f}um, 最大={max_dist:.2f}um")
                prev_min = min_dist
        
        return refined_layers

    def visualize_layer_progress(self, modules: List[PhysicalModule], layer_info: List[List[PhysicalModule]], filename: str = "layer_progress.png", show_guards: bool = True) -> None:
        """可视化分层进度。

        - 绘制当前布局状态；按处理状态着色：已处理层、当前层、未处理层。
        - 若 show_guards=True，则绘制半透明边缘框（spacing guard）。
        - 在图中标注层数与处理状态。
        - 输出到统一结果目录。
        """
        results_dir = "0_layout/code/layout_result"
        os.makedirs(results_dir, exist_ok=True)

        if not modules:
            return

        # 计算边界
        all_x = [m.x for m in modules] + [m.x + m.width for m in modules]
        all_y = [m.y for m in modules] + [m.y + m.height for m in modules]
        x_margin = (max(all_x) - min(all_x)) * 0.1 if all_x else 1.0
        y_margin = (max(all_y) - min(all_y)) * 0.1 if all_y else 1.0
        x_min, x_max = min(all_x) - x_margin, max(all_x) + x_margin
        y_min, y_max = min(all_y) - y_margin, max(all_y) + y_margin

        # 解析当前层索引（从文件名中提取如 layer_2_adjustment.png）
        current_layer_idx = -1
        try:
            import re
            m = re.search(r"layer_(\d+)_", filename)
            if m:
                current_layer_idx = int(m.group(1))
        except Exception:
            current_layer_idx = -1

        num_layers = max(1, len(layer_info))

        # 建立模块到层索引映射
        # 注意：使用模块ID而不是模块对象，因为模块对象可能被复制
        module_to_layer = {}
        for idx, layer in enumerate(layer_info):
            for mod in layer:
                # 使用模块ID作为键，确保映射正确
                module_id = mod.id if hasattr(mod, 'id') else str(mod)
                module_to_layer[module_id] = idx

        # 配色：已处理、当前、未处理
        processed_color = (0.6, 0.6, 0.6, 0.7)   # 灰
        current_color = (1.0, 0.84, 0.0, 0.8)    # 金黄
        pending_color = (0.4, 0.7, 1.0, 0.7)     # 淡蓝
        guard_color = (1.0, 0.75, 0.8, 0.4)      # 边缘框淡粉色，更明显
        guard_edge_color = (1.0, 0.4, 0.6, 0.6)  # 边缘框边框颜色

        fig, ax = plt.subplots(figsize=(8, 6))

        # 先绘制所有边缘框（zorder=1，在底层）
        if show_guards:
            for m in modules:
                if getattr(m, 'spacing_guard_active', False):
                    guard = getattr(m, 'spacing_guard_width', 0.0)
                    # 如果宽度为0但ratio存在，重新计算
                    if guard <= 0:
                        guard_ratio = getattr(m, 'spacing_guard_ratio', 0.0)
                        if guard_ratio > 0:
                            guard = min(m.width, m.height) * guard_ratio
                    
                    if guard > 0:  # 只有当边缘框宽度大于0时才绘制
                        gx = m.x - guard
                        gy = m.y - guard
                        gw = m.width + 2 * guard
                        gh = m.height + 2 * guard
                        # 绘制边缘框填充（半透明）
                        guard_rect = patches.Rectangle(
                            (gx, gy), gw, gh, 
                            linewidth=1.2, 
                            edgecolor=guard_edge_color, 
                            facecolor=guard_color,
                            linestyle='--',  # 虚线边框
                            zorder=1  # 在底层
                        )
                        ax.add_patch(guard_rect)

        # 再绘制所有模块（zorder=2，在上层）
        # 注意：使用模块ID查找层索引，确保使用最新的模块位置
        for m in modules:
            module_id = m.id if hasattr(m, 'id') else str(m)
            layer_idx = module_to_layer.get(module_id, num_layers - 1)
            if current_layer_idx >= 0:
                if layer_idx < current_layer_idx:
                    face = processed_color
                elif layer_idx == current_layer_idx:
                    face = current_color
                else:
                    face = pending_color
            else:
                # 初始图：全部视为未处理
                face = pending_color

            rect = patches.Rectangle((m.x, m.y), m.width, m.height, linewidth=1.5, edgecolor='black', facecolor=face, zorder=2)
            ax.add_patch(rect)
            ax.text(m.x + m.width/2, m.y + m.height/2, m.id, fontsize=6, ha='center', va='center', zorder=3)

        # 计算中心位置（用于显示层次关系）
        if layer_info and len(layer_info) > 0:
            center_macro = layer_info[0][0] if layer_info[0] else None
            if center_macro:
                center_pos = center_macro.get_center()
                # 绘制中心点标记
                ax.plot(center_pos[0], center_pos[1], 'r*', markersize=15, label='中心模块', zorder=100)
                
                # 计算并显示每层到中心的平均距离
                def dist_to_center(m):
                    pos = m.get_center()
                    return math.sqrt((pos[0]-center_pos[0])**2 + (pos[1]-center_pos[1])**2)
                
                layer_distances = []
                for i, layer in enumerate(layer_info):
                    if layer:
                        avg_dist = sum(dist_to_center(m) for m in layer) / len(layer)
                        layer_distances.append(f"第{i}层: {avg_dist:.1f}um")
        
        # 标注处理状态信息（包含距离信息）
        processed_layers = list(range(current_layer_idx)) if current_layer_idx > 0 else []
        pending_layers = list(range(current_layer_idx + 1, num_layers)) if current_layer_idx >= 0 else list(range(num_layers))
        
        status_text = (
            f"层数: {num_layers}\n"
            f"当前层: {current_layer_idx if current_layer_idx >= 0 else '无'}\n"
            f"已处理层: {processed_layers}\n"
            f"未处理层: {pending_layers}"
        )
        
        # 如果有距离信息，添加到状态文本
        if 'layer_distances' in locals() and layer_distances:
            status_text += "\n\n各层平均距离中心:\n" + "\n".join(layer_distances[:5])  # 最多显示5层
        
        ax.text(0.01, 0.99, status_text, transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        # 轴设定
        ax.set_title("BFS分层进度")
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=processed_color, edgecolor='black', label='已处理层'),
            Patch(facecolor=current_color, edgecolor='black', label='当前层'),
            Patch(facecolor=pending_color, edgecolor='black', label='未处理层'),
        ]
        if show_guards:
            # 边缘框图例：使用虚线边框和填充色
            guard_legend = Patch(facecolor=guard_color, edgecolor=guard_edge_color, 
                                linestyle='--', linewidth=1.2, label='边缘框')
            legend_elements.append(guard_legend)
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"层级进度图已保存到 {results_dir}/{filename}")

    def calculate_target_spacing(self, mod1: PhysicalModule, mod2: PhysicalModule) -> float:
        """根据8%/24%规则计算目标间距。

        规则：若任一为敏感/障碍（视作敏感），目标间距=min(min_edge1, min_edge2)*24%；
             否则（数字-数字等）目标间距= min(min_edge1, min_edge2)*8%。
        """
        def is_sensitive(module: PhysicalModule) -> bool:
            if module.type == 'obstacle':
                return True
            return getattr(module, 'sensitivity', 'digital') != 'digital'

        min_edge1 = min(mod1.width, mod1.height)
        min_edge2 = min(mod2.width, mod2.height)
        base_edge = min(min_edge1, min_edge2)

        ratio = 0.24 if (is_sensitive(mod1) or is_sensitive(mod2)) else 0.08
        target = max(0.0, base_edge * ratio)
        return target

    def get_movement_direction(self, current: PhysicalModule, neighbor: PhysicalModule) -> Tuple[float, float]:
        """返回从current指向neighbor的单位方向向量。

        若中心重合，则根据尺寸选择水平优先方向。
        """
        cx, cy = current.get_center()
        nx_, ny_ = neighbor.get_center()
        dx = nx_ - cx
        dy = ny_ - cy
        dist = math.hypot(dx, dy)
        if dist > 1e-12:
            return (dx / dist, dy / dist)

        # 退化情况：中心一致，优先沿较大尺寸轴
        if current.width >= current.height:
            return (1.0, 0.0)
        else:
            return (0.0, 1.0)

    def check_overlap_with_all_modules(self, module: PhysicalModule, all_modules: List[PhysicalModule], exclude_ids: Set[str] = None) -> Tuple[bool, List[str]]:
        """检查模块是否与所有其他模块重叠（排除指定模块）
        
        参数:
            module: 要检查的模块
            all_modules: 所有模块列表
            exclude_ids: 要排除的模块ID集合（通常排除自身）
        
        返回:
            (是否有重叠, 重叠的模块ID列表)
        """
        if exclude_ids is None:
            exclude_ids = {module.id}
        
        overlapping_modules = []
        for other in all_modules:
            if other.id in exclude_ids:
                continue
            overlap, _ = self.check_module_overlap(module, other)
            if overlap:
                overlapping_modules.append(other.id)
        
        return len(overlapping_modules) > 0, overlapping_modules

    def move_module_away(self, module_to_move: PhysicalModule, reference_module: PhysicalModule, target_spacing: float, all_modules: List[PhysicalModule] = None) -> PhysicalModule:
        """推远模块：移动 module_to_move 使其远离 reference_module，达到 target_spacing
        
        当前间距小于目标间距时使用此方法。
        严格检查移动后不与任何模块重叠。
        
        参数:
            module_to_move: 要移动的模块（外层模块）
            reference_module: 参考模块（内层邻居，固定不动）
            target_spacing: 目标边-边间距
            all_modules: 所有模块列表（用于重叠检查）
        
        返回:
            移动后的新模块（如果移动会导致重叠，返回原模块）
        """
        current_gap = self.calculate_module_gap(module_to_move, reference_module)
        
        # 需要增加的边-边距离
        delta = (target_spacing - current_gap)
        if delta <= 0:
            return module_to_move.copy()
        
        delta += 1e-9  # 安全余量
        
        # 获取两个模块的边界
        move_x1, move_x2, move_y1, move_y2 = self._get_effective_bounds(module_to_move)
        ref_x1, ref_x2, ref_y1, ref_y2 = self._get_effective_bounds(reference_module)
        
        # 计算x和y方向的重叠/间距
        x_overlap = min(move_x2, ref_x2) - max(move_x1, ref_x1)
        y_overlap = min(move_y2, ref_y2) - max(move_y1, ref_y1)
        
        new_x = module_to_move.x
        new_y = module_to_move.y
        
        # 如果x方向有重叠，说明模块主要在y方向相邻，沿y方向推远
        if x_overlap > 0:
            move_center_y = (move_y1 + move_y2) / 2
            ref_center_y = (ref_y1 + ref_y2) / 2
            if move_center_y > ref_center_y:
                # 移动模块在上方，向上推
                new_y = module_to_move.y + delta
            else:
                # 移动模块在下方，向下推
                new_y = module_to_move.y - delta
        
        # 如果y方向有重叠，说明模块主要在x方向相邻，沿x方向推远
        elif y_overlap > 0:
            move_center_x = (move_x1 + move_x2) / 2
            ref_center_x = (ref_x1 + ref_x2) / 2
            if move_center_x > ref_center_x:
                # 移动模块在右侧，向右推
                new_x = module_to_move.x + delta
            else:
                # 移动模块在左侧，向左推
                new_x = module_to_move.x - delta
        
        # 对角关系：沿中心连线方向推，但需要考虑投影
        else:
            push_dir = self.get_movement_direction(reference_module, module_to_move)
            # 对角情况下，移动距离需要根据方向进行调整
            # 为了使边-边距离增加delta，需要沿对角线移动更多
            move_magnitude = delta * 1.5  # 经验系数
            new_x = module_to_move.x + push_dir[0] * move_magnitude
            new_y = module_to_move.y + push_dir[1] * move_magnitude
        
        # 边界检查
        layout_w = getattr(self, 'layout_width', None)
        layout_h = getattr(self, 'layout_height', None)
        if layout_w is not None and layout_h is not None and layout_w > 0 and layout_h > 0:
            new_x = max(0.0, min(layout_w - module_to_move.width, new_x))
            new_y = max(0.0, min(layout_h - module_to_move.height, new_y))
        
        # 创建移动后的临时模块
        temp_module = module_to_move.copy(new_x=new_x, new_y=new_y)
        
        # 严格检查：移动后是否与任何模块重叠
        if all_modules is not None:
            has_overlap, overlapping_ids = self.check_overlap_with_all_modules(
                temp_module, all_modules, exclude_ids={module_to_move.id}
            )
            if has_overlap:
                # 如果会导致重叠，尝试渐进式移动（减少移动距离）
                max_attempts = 5
                for attempt in range(1, max_attempts + 1):
                    scale = 1.0 - (attempt * 0.2)  # 逐步减少移动距离：0.8, 0.6, 0.4, 0.2, 0.0
                    if scale <= 0:
                        break
                    
                    scaled_delta = delta * scale
                    if x_overlap > 0:
                        if move_center_y > ref_center_y:
                            new_y = module_to_move.y + scaled_delta
                        else:
                            new_y = module_to_move.y - scaled_delta
                    elif y_overlap > 0:
                        if move_center_x > ref_center_x:
                            new_x = module_to_move.x + scaled_delta
                        else:
                            new_x = module_to_move.x - scaled_delta
                    else:
                        new_x = module_to_move.x + push_dir[0] * scaled_delta * 1.5
                        new_y = module_to_move.y + push_dir[1] * scaled_delta * 1.5
                    
                    # 边界检查
                    if layout_w is not None and layout_h is not None and layout_w > 0 and layout_h > 0:
                        new_x = max(0.0, min(layout_w - module_to_move.width, new_x))
                        new_y = max(0.0, min(layout_h - module_to_move.height, new_y))
                    
                    temp_module = module_to_move.copy(new_x=new_x, new_y=new_y)
                    has_overlap, overlapping_ids = self.check_overlap_with_all_modules(
                        temp_module, all_modules, exclude_ids={module_to_move.id}
                    )
                    if not has_overlap:
                        # 找到不重叠的位置
                        return temp_module
                
                # 所有尝试都失败，返回原模块
                return module_to_move.copy()
        
        return temp_module

    def move_module_closer(self, module_to_move: PhysicalModule, reference_module: PhysicalModule, target_spacing: float, all_modules: List[PhysicalModule] = None) -> PhysicalModule:
        """拉近模块：移动 module_to_move 使其靠近 reference_module，达到 target_spacing
        
        当前间距大于目标间距时使用此方法。
        严格检查移动后不与任何模块重叠。
        
        参数:
            module_to_move: 要移动的模块（外层模块）
            reference_module: 参考模块（内层邻居，固定不动）
            target_spacing: 目标边-边间距
            all_modules: 所有模块列表（用于重叠检查）
        
        返回:
            移动后的新模块（如果移动会导致重叠，返回原模块）
        """
        current_gap = self.calculate_module_gap(module_to_move, reference_module)
        
        # 需要减少的边-边距离
        delta = (current_gap - target_spacing)
        if delta <= 0:
            return module_to_move.copy()
        
        delta -= 1e-9  # 留一点余量，避免过度拉近
        delta = max(0, delta)
        
        # 获取两个模块的边界
        move_x1, move_x2, move_y1, move_y2 = self._get_effective_bounds(module_to_move)
        ref_x1, ref_x2, ref_y1, ref_y2 = self._get_effective_bounds(reference_module)
        
        # 计算x和y方向的间距（注意：这里是间距，不是重叠）
        x_gap = max(ref_x1 - move_x2, move_x1 - ref_x2)
        y_gap = max(ref_y1 - move_y2, move_y1 - ref_y2)
        
        new_x = module_to_move.x
        new_y = module_to_move.y
        
        # 如果主要是在x方向分离，沿x方向拉近
        if x_gap > 0 and x_gap >= y_gap:
            move_center_x = (move_x1 + move_x2) / 2
            ref_center_x = (ref_x1 + ref_x2) / 2
            if move_center_x > ref_center_x:
                # 移动模块在右侧，向左拉
                new_x = module_to_move.x - delta
            else:
                # 移动模块在左侧，向右拉
                new_x = module_to_move.x + delta
        
        # 如果主要是在y方向分离，沿y方向拉近
        elif y_gap > 0 and y_gap > x_gap:
            move_center_y = (move_y1 + move_y2) / 2
            ref_center_y = (ref_y1 + ref_y2) / 2
            if move_center_y > ref_center_y:
                # 移动模块在上方，向下拉
                new_y = module_to_move.y - delta
            else:
                # 移动模块在下方，向上拉
                new_y = module_to_move.y + delta
        
        # 对角情况或已经有重叠但间距还不够（可能是对角接触）
        else:
            pull_dir = self.get_movement_direction(module_to_move, reference_module)
            move_magnitude = delta * 1.5  # 经验系数
            new_x = module_to_move.x + pull_dir[0] * move_magnitude
            new_y = module_to_move.y + pull_dir[1] * move_magnitude
        
        # 边界检查
        layout_w = getattr(self, 'layout_width', None)
        layout_h = getattr(self, 'layout_height', None)
        if layout_w is not None and layout_h is not None and layout_w > 0 and layout_h > 0:
            new_x = max(0.0, min(layout_w - module_to_move.width, new_x))
            new_y = max(0.0, min(layout_h - module_to_move.height, new_y))
        
        # 创建移动后的临时模块
        temp_module = module_to_move.copy(new_x=new_x, new_y=new_y)
        
        # 严格检查：移动后是否与任何模块重叠
        if all_modules is not None:
            has_overlap, overlapping_ids = self.check_overlap_with_all_modules(
                temp_module, all_modules, exclude_ids={module_to_move.id}
            )
            if has_overlap:
                # 如果会导致重叠，尝试渐进式移动（减少移动距离）
                max_attempts = 5
                for attempt in range(1, max_attempts + 1):
                    scale = 1.0 - (attempt * 0.2)  # 逐步减少移动距离：0.8, 0.6, 0.4, 0.2, 0.0
                    if scale <= 0:
                        break
                    
                    scaled_delta = delta * scale
                    if x_gap > 0 and x_gap >= y_gap:
                        if move_center_x > ref_center_x:
                            new_x = module_to_move.x - scaled_delta
                        else:
                            new_x = module_to_move.x + scaled_delta
                    elif y_gap > 0 and y_gap > x_gap:
                        if move_center_y > ref_center_y:
                            new_y = module_to_move.y - scaled_delta
                        else:
                            new_y = module_to_move.y + scaled_delta
                    else:
                        new_x = module_to_move.x + pull_dir[0] * scaled_delta * 1.5
                        new_y = module_to_move.y + pull_dir[1] * scaled_delta * 1.5
                    
                    # 边界检查
                    if layout_w is not None and layout_h is not None and layout_w > 0 and layout_h > 0:
                        new_x = max(0.0, min(layout_w - module_to_move.width, new_x))
                        new_y = max(0.0, min(layout_h - module_to_move.height, new_y))
                    
                    temp_module = module_to_move.copy(new_x=new_x, new_y=new_y)
                    has_overlap, overlapping_ids = self.check_overlap_with_all_modules(
                        temp_module, all_modules, exclude_ids={module_to_move.id}
                    )
                    if not has_overlap:
                        # 找到不重叠的位置
                        return temp_module
                
                # 所有尝试都失败，返回原模块
                return module_to_move.copy()
        else:
            # 如果没有提供all_modules，至少检查与参考模块的重叠
            overlap, _ = self.check_module_overlap(temp_module, reference_module)
            if overlap:
                return module_to_move.copy()
        
        return temp_module

    def move_module_to_target_spacing(self, inner_mod: PhysicalModule, outer_mod: PhysicalModule, target_spacing: float) -> PhysicalModule:
        """移动 inner_mod 使其与 outer_mod 的边-边间距达到 target_spacing。

        - 若当前间距 >= 目标，不移动
        - 否则沿远离 outer_mod 的方向移动所需距离
        - 移动后进行边界检查，限定在 [0, layout_width - w] x [0, layout_height - h]
        返回移动后的新模块（不原地修改传入对象）
        """
        # 当前间距（>0=间隙，<0=重叠深度）
        current_gap = self.calculate_module_gap(inner_mod, outer_mod)

        if current_gap >= target_spacing - 1e-12:
            return inner_mod.copy()  # 不需要移动

        # 方向：从 outer 指向 inner（即 inner 远离 outer）
        dir_to_outer = self.get_movement_direction(inner_mod, outer_mod)
        move_dir = (-dir_to_outer[0], -dir_to_outer[1])

        # 需要增加的边-边距离
        delta = (target_spacing - current_gap)
        # 安全余量，避免数值误差
        delta = max(0.0, delta) + 1e-9

        # 将边-边距离变化映射为位移：沿中心连线方向平移
        new_x = inner_mod.x + move_dir[0] * delta
        new_y = inner_mod.y + move_dir[1] * delta

        # 边界检查（使用已配置的布局范围）
        layout_w = getattr(self, 'layout_width', None)
        layout_h = getattr(self, 'layout_height', None)
        if layout_w is not None and layout_h is not None and layout_w > 0 and layout_h > 0:
            new_x = max(0.0, min(layout_w - inner_mod.width, new_x))
            new_y = max(0.0, min(layout_h - inner_mod.height, new_y))

        return inner_mod.copy(new_x=new_x, new_y=new_y)

    def run_agent_scaling_experiment(self, agent_counts: List[int] = [1, 2, 3, 4, 5]) -> Dict[int, float]:
        """运行不同智能体数量的布局优化实验，返回每个代理数量的最佳HPWL，支持紧凑"""
        print("\n=== 智能体缩放实验开始（紧凑模式）===")

        # 计算原始布局的基线HPWL
        # 将original_positions转换为模块中心坐标
        original_center_positions = {}
        for module_id, (x, y) in self.original_positions.items():
            if module_id in self.modules:
                module = self.modules[module_id]
                center_x = x + module.width / 2
                center_y = y + module.height / 2
                original_center_positions[module_id] = (center_x, center_y)
        
        baseline_hpwl = self._calculate_layout_hpwl(original_center_positions)
        print(f"原始布局基线HPWL: {baseline_hpwl:.2f}")

        results = {0: baseline_hpwl}  # 0代理表示原始

        for num_agents in agent_counts:
            print(f"\n--- 测试 {num_agents} 个智能体 ---")

            # 创建临时优化器实例
            temp_modules = [PhysicalModule(m.id, m.x, m.y, m.width, m.height, m.type, m.layer, m.pins.copy())
                            for m in self.modules.values()]
            temp_optimizer = LayoutOptimizer(
                temp_modules,
                self.connections,
                num_agents=num_agents,
                min_spacing=self.min_spacing
            )

            # 运行完整优化（包括吸引）
            temp_optimized = temp_optimizer.optimize_layout(num_runs_per_agent=3)

            # 计算吸引后的HPWL
            temp_positions = {m.id: m.get_center() for m in temp_optimized}
            best_hpwl = temp_optimizer._calculate_layout_hpwl(temp_positions)

            results[num_agents] = best_hpwl
            improvement = ((baseline_hpwl - best_hpwl) / baseline_hpwl * 100) if baseline_hpwl > 0 else 0
            print(f"{num_agents} 个智能体紧凑HPWL: {best_hpwl:.2f} (改进: {improvement:.2f}%)")

        print("\n=== 实验完成 ===")
        return results

    def _get_module_color(self, module_type: str) -> str:
        """根据模块类型获取颜色"""
        color_map = {
            'clk': '#FF6B6B',  # 红色
            'analog': '#FFD700',  # 黄色
            'digital': '#45B7D1',  # 蓝色
            'memory': '#FFA07A',  # 橙色
            'io': '#98D8C8',  # 青色
            'obstacle': '#888888'  # 灰色
        }
        return color_map.get(module_type, '#FFFFFF')  # 默认白色

    def _add_legend(self, ax):
        """添加图例"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='black', label='时钟模块'),
            Patch(facecolor='salmon', edgecolor='black', label='模拟模块'),
            Patch(facecolor='lightblue', edgecolor='black', label='数字模块'),
            Patch(facecolor='lightyellow', edgecolor='black', label='存储模块'),
            Patch(facecolor='lavender', edgecolor='black', label='IO模块'),
            Patch(facecolor='lightgray', edgecolor='black', label='障碍物')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def analyze_gap_statistics(self, modules):
        """分析模块间缝隙统计信息"""
        gaps = []
        overlaps = []

        for i in range(len(modules)):
            mod_i = modules[i]
            for j in range(i + 1, len(modules)):
                mod_j = modules[j]
                gap = self.calculate_module_gap(mod_i, mod_j)

                if gap < 0:
                    overlaps.append(-gap)  # 重叠深度
                else:
                    gaps.append(gap)

        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            min_gap = min(gaps)
            max_gap = max(gaps)
            print(f"缝隙统计: 平均={avg_gap:.2f}um, 最小={min_gap:.2f}um, 最大={max_gap:.2f}um")

        if overlaps:
            avg_overlap = sum(overlaps) / len(overlaps)
            max_overlap = max(overlaps)
            print(f"重叠统计: 平均重叠={avg_overlap:.2f}um, 最大重叠={max_overlap:.2f}um")

        return {
            'gaps': gaps,
            'overlaps': overlaps,
            'avg_gap': sum(gaps) / len(gaps) if gaps else 0,
            'min_gap': min(gaps) if gaps else float('inf'),
            'max_gap': max(gaps) if gaps else 0
        }

    def check_spacing(self, modules):
        """检查所有模块是否满足最小间距要求"""
        all_good = True
        overlap_count = 0

        for i in range(len(modules)):
            mod_i = modules[i]

            for j in range(i + 1, len(modules)):
                mod_j = modules[j]

                # 使用严格的重叠检查方法
                overlap, overlap_area = self.check_module_overlap(mod_i, mod_j)
                if overlap:
                    overlap_count += 1
                    print(f"[警告] 严重警告：模块 {mod_i.id} 和 {mod_j.id} 发生重叠，重叠面积={overlap_area:.6f}")
                    all_good = False
                    continue

                # 计算实际间距
                gap = self.calculate_module_gap(mod_i, mod_j)

                # 如果间距小于最小值，记录问题
                if gap < self.min_spacing:
                    print(
                        f"警告：模块 {mod_i.id} 和 {mod_j.id} 间距不足 (当前: {gap:.2f}um, 要求: {self.min_spacing}um)")
                    all_good = False

        if overlap_count > 0:
            print(f"发现 {overlap_count} 个重叠问题")

        return all_good
    
    # ========== 阶段2：全局DRC检查与报告 ==========
    
    def check_drc_violations(self, modules: List[PhysicalModule]) -> Dict[str, Any]:
        """
        执行全局DRC检查，检查所有模块对之间的间距违规
        
        参数:
            modules: 模块列表
        
        返回:
            包含违规统计信息的字典
        """
        violations = []  # 存储所有违规信息
        total_pairs = 0
        
        # 统计不同类型的模块对
        digital_digital_pairs = 0
        sensitive_sensitive_pairs = 0
        digital_sensitive_pairs = 0
        
        # 遍历所有模块对
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                mod_i = modules[i]
                mod_j = modules[j]
                total_pairs += 1
                
                # 检查DRC违规
                is_violation, required_spacing, actual_distance = mod_i.check_drc_violation(mod_j)
                
                # 统计模块对类型
                if mod_i.sensitivity == 'digital' and mod_j.sensitivity == 'digital':
                    digital_digital_pairs += 1
                elif mod_i.sensitivity == 'sensitive' and mod_j.sensitivity == 'sensitive':
                    sensitive_sensitive_pairs += 1
                else:
                    digital_sensitive_pairs += 1
                
                # 如果存在违规，记录详细信息
                if is_violation:
                    violation_info = {
                        'module_a': mod_i.id,
                        'module_b': mod_j.id,
                        'type_a': mod_i.type,
                        'type_b': mod_j.type,
                        'sensitivity_a': mod_i.sensitivity,
                        'sensitivity_b': mod_j.sensitivity,
                        'required_spacing': required_spacing,
                        'actual_distance': actual_distance,
                        'violation_amount': required_spacing - actual_distance
                    }
                    violations.append(violation_info)
        
        # 汇总统计
        summary = {
            'total_pairs': total_pairs,
            'total_violations': len(violations),
            'violation_rate': (len(violations) / total_pairs * 100) if total_pairs > 0 else 0,
            'digital_digital_pairs': digital_digital_pairs,
            'sensitive_sensitive_pairs': sensitive_sensitive_pairs,
            'digital_sensitive_pairs': digital_sensitive_pairs,
            'violations': violations
        }
        
        return summary
    
    def print_drc_report(self, modules: List[PhysicalModule], verbose: bool = True):
        """
        打印DRC检查报告
        
        参数:
            modules: 模块列表
            verbose: 是否打印详细的违规信息
        """
        print("\n" + "=" * 60)
        print("          DRC 间距规则检查报告")
        print("=" * 60)
        
        # 执行DRC检查
        drc_summary = self.check_drc_violations(modules)
        
        # 打印基本统计
        print(f"\n【总体统计】")
        print(f"  总模块对数: {drc_summary['total_pairs']}")
        print(f"  DRC违规数量: {drc_summary['total_violations']}")
        print(f"  违规率: {drc_summary['violation_rate']:.2f}%")
        
        print(f"\n【模块对类型分布与规则】")
        print(f"  数字宏 <-> 数字宏: {drc_summary['digital_digital_pairs']} 对 (规则: 8%最短边)")
        print(f"  敏感宏 <-> 敏感宏: {drc_summary['sensitive_sensitive_pairs']} 对 (规则: 24%最短边)")
        print(f"  数字宏 <-> 敏感宏: {drc_summary['digital_sensitive_pairs']} 对 (规则: 24%最短边)")
        
        # 如果有违规且verbose=True，打印详细信息
        if drc_summary['total_violations'] > 0:
            print(f"\n【违规详情】")
            if verbose:
                for idx, v in enumerate(drc_summary['violations'][:10], 1):  # 最多显示10条
                    print(f"\n  违规 #{idx}:")
                    print(f"    模块对: {v['module_a']} ({v['sensitivity_a']}) ↔ {v['module_b']} ({v['sensitivity_b']})")
                    print(f"    要求间距: {v['required_spacing']:.2f}μm")
                    print(f"    实际间距: {v['actual_distance']:.2f}μm")
                    print(f"    违规量: {v['violation_amount']:.2f}μm")
                
                if drc_summary['total_violations'] > 10:
                    print(f"\n  ... 还有 {drc_summary['total_violations'] - 10} 条违规未显示")
            else:
                print(f"  (详细信息已隐藏，设置 verbose=True 查看)")
        else:
            print(f"\n✅ 恭喜！所有模块均满足DRC间距要求！")
        
        print("\n" + "=" * 60)
        
        return drc_summary

    def visualize_drc_violations(self, modules: List[PhysicalModule], drc_summary: Dict[str, Any], filename: str = "drc_violations.png") -> None:
        """可视化DRC检查结果：
        - 绘制所有模块矩形
        - 对违规的模块对，在中心连线处画红线，线宽与违规量成比例
        - 标注违规量（μm）
        """
        results_dir = "0_layout/code/layout_result"
        os.makedirs(results_dir, exist_ok=True)

        if not modules:
            return

        # 坐标边界
        all_x = [m.x for m in modules] + [m.x + m.width for m in modules]
        all_y = [m.y for m in modules] + [m.y + m.height for m in modules]
        x_margin = (max(all_x) - min(all_x)) * 0.1 if all_x else 1.0
        y_margin = (max(all_y) - min(all_y)) * 0.1 if all_y else 1.0
        x_min, x_max = min(all_x) - x_margin, max(all_x) + x_margin
        y_min, y_max = min(all_y) - y_margin, max(all_y) + y_margin

        # 建立ID索引
        id_to_module = {m.id: m for m in modules}

        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制模块
        for m in modules:
            color = self._get_module_color(m.type)
            rect = patches.Rectangle((m.x, m.y), m.width, m.height, linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            ax.text(m.x + m.width/2, m.y + m.height/2, m.id, fontsize=7, ha='center', va='center')

        # 绘制违规连线
        violations = drc_summary.get('violations', []) if isinstance(drc_summary, dict) else []
        if violations:
            max_violation = max(v.get('violation_amount', 0.0) for v in violations) or 1.0
            for v in violations:
                a, b = v['module_a'], v['module_b']
                if a in id_to_module and b in id_to_module:
                    ma, mb = id_to_module[a], id_to_module[b]
                    ax_a = ma.get_center()
                    ax_b = mb.get_center()
                    amount = max(0.0, v.get('violation_amount', 0.0))
                    lw = 0.5 + 3.5 * (amount / max_violation)
                    ax.plot([ax_a[0], ax_b[0]], [ax_a[1], ax_b[1]], color='red', linewidth=lw, alpha=0.8)
                    # 在线段中点标注违规量
                    midx, midy = (ax_a[0] + ax_b[0]) / 2, (ax_a[1] + ax_b[1]) / 2
                    ax.text(midx, midy, f"{amount:.2f}μm", color='red', fontsize=7, ha='center', va='bottom')

        # 轴设置
        ax.set_title("DRC违规可视化（红线越粗表示违规越严重）")
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)

        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='DRC违规连线'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(results_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"DRC违规可视化图已保存到 {out_path}")

    def calculate_forces(self):
        """计算所有模块间的力学作用力"""

        # 重置所有模块的受力
        for module in self.modules.values():
            module.reset_forces()

        module_list = list(self.modules.values())

        # 1. 计算库仑斥力（所有模块对之间）
        for i, mod1 in enumerate(module_list):
            for j, mod2 in enumerate(module_list[i + 1:], i + 1):
                if mod1.is_fixed and mod2.is_fixed:
                    continue  # 固定模块之间不计算斥力

                # 计算模块中心距离
                x1, y1 = mod1.get_center()
                x2, y2 = mod2.get_center()
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < 1e-6:  # 避免除零
                    distance = 1e-6

                # 库仑斥力：F = k * (q1 * q2) / r^2
                force_magnitude = self.coulomb_constant * (mod1.charge * mod2.charge) / (distance * distance)

                # 力的方向（从mod1指向mod2）
                fx = force_magnitude * dx / distance
                fy = force_magnitude * dy / distance

                # 应用斥力（mod1受到远离mod2的力，mod2受到远离mod1的力）
                mod1.apply_force(-fx, -fy)
                mod2.apply_force(fx, fy)

        # 2. 计算MST连接间的弹簧引力
        for src_id, _, dst_id, _ in self.connections:
            if src_id not in self.modules or dst_id not in self.modules:
                continue

            mod1 = self.modules[src_id]
            mod2 = self.modules[dst_id]

            # 计算模块中心距离
            x1, y1 = mod1.get_center()
            x2, y2 = mod2.get_center()
            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < 1e-6:
                distance = 1e-6

            # 计算理想长度（基于模块大小和最小间距）
            ideal_length = (mod1.width + mod2.width) / 2 + (mod1.height + mod2.height) / 2 + self.min_spacing

            # 弹簧力：F = k * (r - ideal_length)
            spring_force = (mod1.spring_constant + mod2.spring_constant) / 2 * (distance - ideal_length)

            # 力的方向（从mod1指向mod2）
            fx = spring_force * dx / distance
            fy = spring_force * dy / distance

            # 应用弹簧力
            mod1.apply_force(fx, fy)
            mod2.apply_force(-fx, -fy)

    def update_positions(self):
        """根据力学作用更新模块位置"""
        for module in self.modules.values():
            if module.is_fixed:
                continue  # 固定模块不移动

            # 根据F=ma计算加速度
            if module.mass > 0:
                ax = module.force_x / module.mass
                ay = module.force_y / module.mass

                # 更新速度：v = v + a*dt
                module.velocity_x += ax * self.time_step
                module.velocity_y += ay * self.time_step

                # 应用速度限制
                velocity_magnitude = math.sqrt(module.velocity_x ** 2 + module.velocity_y ** 2)
                if velocity_magnitude > self.max_velocity:
                    scale = self.max_velocity / velocity_magnitude
                    module.velocity_x *= scale
                    module.velocity_y *= scale

                # 应用阻尼
                module.velocity_x *= (1 - self.damping)
                module.velocity_y *= (1 - self.damping)

                # 更新位置：p = p + v*dt
                module.x += module.velocity_x * self.time_step
                module.y += module.velocity_y * self.time_step

    def cool_down(self):
        """模拟退火冷却，逐渐减小力的强度"""
        self.iteration += 1

        # 冷却因子：随着迭代进行逐渐减小
        cool_factor = 1.0 / (1.0 + 0.1 * self.iteration)

        # 减小库仑常数和弹簧常数
        self.coulomb_constant *= cool_factor
        self.damping = min(0.9, self.damping + 0.01)  # 逐渐增加阻尼

        # 减小时间步长（更稳定的模拟）
        self.time_step = max(0.01, self.time_step * 0.99)

    def mechanical_simulation_step(self):
        """执行一步力学模拟"""
        self.calculate_forces()
        self.update_positions()
        self.cool_down()

    def get_system_energy(self) -> float:
        """计算系统总能量（动能 + 势能）"""
        total_kinetic = sum(module.get_kinetic_energy() for module in self.modules.values())

        # 简化的势能计算（基于模块间距离）
        total_potential = 0.0
        module_list = list(self.modules.values())
        for i, mod1 in enumerate(module_list):
            for j, mod2 in enumerate(module_list[i + 1:], i + 1):
                x1, y1 = mod1.get_center()
                x2, y2 = mod2.get_center()
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance > 0:
                    total_potential += self.coulomb_constant * (mod1.charge * mod2.charge) / distance

        return total_kinetic + total_potential

    def attract_modules(self, modules, attract_factor):
        """超紧凑模块拉近：最大化模块密度，但严格禁止重叠"""
        print(f"超紧凑模块拉近（吸引因子={attract_factor}），严格禁止重叠...")

        # 计算模块中心
        centers = [module.get_center() for module in modules]
        center_x = sum(center[0] for center in centers) / len(modules)
        center_y = sum(center[1] for center in centers) / len(modules)

        # 创建模块副本用于安全调整
        module_list = [module.copy() for module in modules]

        # 多轮拉近，逐步增加吸引强度
        max_rounds = 5
        for round_idx in range(max_rounds):
            round_attract_factor = attract_factor * (0.2 + 0.8 * round_idx / max_rounds)  # 渐进式增加
            print(f"拉近轮次 {round_idx + 1}/{max_rounds}，吸引因子={round_attract_factor:.2f}")

            modules_moved = 0

            # 调整模块位置，但检查重叠
            for i, module in enumerate(module_list):
                if module.type == 'obstacle':
                    continue

                # 计算向中心的移动
                dx = center_x - module.x
                dy = center_y - module.y
                distance = math.sqrt(dx ** 2 + dy ** 2)

                if distance > 0:
                    # 计算移动步长（更激进的移动）
                    move_distance = distance * round_attract_factor * 0.2  # 更大的步长
                    new_x = module.x + (dx / distance) * move_distance
                    new_y = module.y + (dy / distance) * move_distance

                    # 创建临时模块检查重叠
                    temp_module = module.copy(new_x=new_x, new_y=new_y)

                    # 检查是否会产生重叠
                    has_overlap = False
                    for j, other_module in enumerate(module_list):
                        if i != j and other_module.type != 'obstacle':
                            overlap, _ = self.check_module_overlap(temp_module, other_module)
                            if overlap:
                                has_overlap = True
                                break

                    # 如果没有重叠，应用新位置
                    if not has_overlap:
                        module.x = new_x
                        module.y = new_y
                        modules_moved += 1

            print(f"轮次 {round_idx + 1} 完成，移动了 {modules_moved} 个模块")

            # 如果这一轮没有模块移动，提前结束
            if modules_moved == 0:
                break

        # 最终重叠检查
        overlap_count = 0
        total_overlap_area = 0.0
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                overlap, overlap_area = self.check_module_overlap(module_list[i], module_list[j])
                if overlap:
                    overlap_count += 1
                    total_overlap_area += overlap_area

        if overlap_count > 0:
            print(f"[警告] 超紧凑拉近后仍有 {overlap_count} 个重叠，总重叠面积={total_overlap_area:.6f}")
        else:
            print("[成功] 超紧凑拉近后无重叠，分布极度紧凑")

        return module_list




# ============================================================================
# 工具函数
# ============================================================================

def parse_json_design_test5(json_file_path):
    """解析test_5的JSON设计文件，包含MACRO_PIN类型"""
    import json

    with open(json_file_path, 'r', encoding='utf-8') as f:
        design_data = json.load(f)

    modules = []
    connections = []

    # 解析模块信息
    for node in design_data['nodes']:
        node_name = node['name']
        attr = node['attr']

        # 只处理实际的模块，跳过引脚节点
        if attr['type'] in ['MACRO', 'PORT']:
            # 确定模块类型
            if attr['type'] == 'PORT':
                module_type = 'io'
            else:
                # 根据模块名称或其他属性确定类型
                if 'Grp' in node_name:
                    module_type = 'digital'
                elif 'M' in node_name:
                    module_type = 'memory'
                else:
                    module_type = 'digital'

            # 创建引脚列表
            width = attr.get('width', 50.0)
            height = attr.get('height', 50.0)

            if attr['type'] == 'MACRO':
                # 生成4个角点引脚
                pins = [
                    (0, 0),  # 左下角
                    (width, 0),  # 右下角
                    (width, height),  # 右上角
                    (0, height)  # 左上角
                ]
            else:
                # 端口只有一个引脚在中心
                pins = [(width / 2, height / 2)]

            # 读取模块敏感度（如果JSON中提供）
            # 支持的字段名：'sensitivity', 'module_sensitivity', 'macro_type'
            sensitivity = attr.get('sensitivity') or attr.get('module_sensitivity') or attr.get('macro_type')
            
            # 如果JSON中没有提供，则根据module_type推断
            if sensitivity not in ['digital', 'sensitive']:
                sensitivity = None  # 让PhysicalModule自动推断

            module = PhysicalModule(
                module_id=node_name,
                x=attr['x'],
                y=attr['y'],
                width=width,
                height=height,
                module_type=module_type,
                layer=attr.get('layer', 0),
                pins=pins,
                sensitivity=sensitivity  # 传递敏感度信息
            )
            modules.append(module)

    # 解析连接关系 - 通过MACRO_PIN建立连接
    module_names = set(module.id for module in modules)

    # 为了确保有足够的连接进行测试，添加一些基本连接
    if len(modules) >= 2:
        valid_modules = [m for m in modules if m.type != 'obstacle']
        if len(valid_modules) >= 2:
            for i in range(len(valid_modules) - 1):
                src_mod = valid_modules[i]
                dst_mod = valid_modules[i + 1]
                connections.append((src_mod.id, 0, dst_mod.id, 0))

    # 解析JSON中定义的连接（通过MACRO_PIN）
    for node in design_data['nodes']:
        node_name = node['name']
        attr = node['attr']

        # 处理MACRO_PIN类型的连接
        if attr['type'] == 'MACRO_PIN':
            macro_name = attr.get('macro_name')
            if macro_name in module_names:
                inputs = node.get('input', [])
                for input_name in inputs:
                    # 查找输入是否是模块
                    if input_name in module_names:
                        connection = (input_name, 0, macro_name, 0)
                        if connection not in connections:
                            connections.append(connection)

    return modules, connections


def export_layout_to_json(modules: List[PhysicalModule], output_file: str):
    """导出布局结果为JSON格式"""
    data = {
        'modules': [],
        'statistics': {
            'total_modules': len(modules),
            'module_types': {},
            'sensitivity_types': {}
        }
    }
    
    for module in modules:
        mod_data = {
            'id': module.id,
            'x': float(module.x),
            'y': float(module.y),
            'width': float(module.width),
            'height': float(module.height),
            'type': module.type,
            'sensitivity': module.sensitivity,  # 新增：敏感度信息
            'spacing_guard_width': float(module.spacing_guard_width),  # 新增：边缘框宽度
            'layer': module.layer,
            'pins': module.pins
        }
        data['modules'].append(mod_data)
        
        # 统计模块类型
        if module.type not in data['statistics']['module_types']:
            data['statistics']['module_types'][module.type] = 0
        data['statistics']['module_types'][module.type] += 1
        
        # 统计敏感度类型
        if module.sensitivity not in data['statistics']['sensitivity_types']:
            data['statistics']['sensitivity_types'][module.sensitivity] = 0
        data['statistics']['sensitivity_types'][module.sensitivity] += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"布局结果已导出到: {output_file}")
    print(f"  - 数字宏: {data['statistics']['sensitivity_types'].get('digital', 0)} 个")
    print(f"  - 敏感宏: {data['statistics']['sensitivity_types'].get('sensitive', 0)} 个")



# ============================================================================
# 主程序
# ============================================================================

def main():
    """布局优化主程序 - 仅布局优化功能"""
    print("=" * 70)
    print("  芯片布局优化系统 - 多智能体竞争算法")
    print("=" * 70)
    
    # 确定脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建输出目录（在脚本同级目录下）
    output_dir = os.path.join(script_dir, "layout_result")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # ========================================
    # 第1步：读取输入设计
    # ========================================
    print("\n[步骤 1/4] 读取输入设计文件...")
    
    # 获取项目根目录（0_layout目录）
    project_root = os.path.dirname(script_dir)
    
    # 尝试多个可能的输入文件路径（使用绝对路径）
    possible_paths = [
        os.path.join(script_dir, "input1.json"),                # 与脚本同目录
        os.path.join(os.getcwd(), "input1.json"),               # 当前工作目录
        "input1.json",                                          # 相对路径（脚本/工作目录）
        os.path.join(project_root, "input", "input1.json"),  # 标准路径
        os.path.join(script_dir, "..", "input", "input1.json"),  # 从code目录回到根目录
        "../input/input1.json",         # 相对路径（从code目录）
        "./input/input1.json",          # 相对路径（从根目录）
        "input/input1.json",            # 相对路径（从根目录）
        "../../代码/input1.json",       # 旧路径（兼容）
        "./0_layout/input/input1.json"  # 旧路径（兼容）
    ]
    
    input_file = None
    for path in possible_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(path):
            input_file = path
            print(f"找到输入文件: {full_path}")
            break
    
    if input_file is None:
        print("错误: 找不到input1.json文件")
        print("尝试过的路径:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        print(f"\n项目根目录: {project_root}")
        print(f"脚本目录: {script_dir}")
        return
    
    try:
        modules, connections = parse_json_design_test5(input_file)
        print(f"成功加载: {len(modules)} 个模块, {len(connections)} 个连接")
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存原始布局
    original_modules = [
        PhysicalModule(
            module_id=m.id,
            x=m.x,
            y=m.y,
            width=m.width,
            height=m.height,
            module_type=m.type,
            layer=m.layer,
            pins=m.pins.copy()
        ) for m in modules
    ]
    
    # ========================================
    # 第2步：创建布局优化器
    # ========================================
    print("\n[步骤 2/4] 初始化布局优化器...")
    optimizer = LayoutOptimizer(
        modules=modules,
        connections=connections,
        num_agents=5,
        min_spacing=15.0
    )
    
    # ========================================
    # 第3步：运行布局优化
    # ========================================
    print("\n[步骤 3/4] 执行多智能体布局优化...")
    print("-" * 70)
    optimized_modules = optimizer.optimize_layout(
        max_iterations=10,
        num_runs_per_agent=3
    )
    
    # ========================================
    # 第4步：保存和可视化结果
    # ========================================
    print("\n[步骤 4/4] 保存结果和生成可视化...")
    
    # 导出JSON
    json_file = os.path.join(output_dir, "optimized_layout.json")
    export_layout_to_json(optimized_modules, json_file)
    
    # 布局对比可视化
    print("\n生成布局对比图...")
    optimizer.visualize_layout_comparison(original_modules, optimized_modules)
    
    # 竞赛结果可视化
    print("\n生成竞赛结果图...")
    optimizer.visualize_competition_results()
    
    # 间隙分析可视化
    print("\n生成间隙分析图...")
    optimizer.visualize_gap_analysis(original_modules, optimized_modules)
    
    # ========================================
    # 最终统计
    # ========================================
    print("\n" + "=" * 70)
    print("  布局优化完成！")
    print("=" * 70)
    print(f"\n所有结果已保存到目录: {output_dir}/")
    print(f"  ├─ optimized_layout.json                   (布局坐标数据)")
    print(f"  ├─ layout_optimization_comparison.png      (布局对比图)")
    print(f"  ├─ layout_competition_results.png          (竞赛结果图)")
    print(f"  └─ gap_analysis_comparison.png             (间隙分析图)")
    
    # 显示优化效果统计
    print(f"\n优化效果统计:")
    print(f"  - 总模块数: {len(optimized_modules)}")
    print(f"  - 总连接数: {len(connections)}")
    print(f"  - 智能体数量: 5")
    print(f"  - 优化迭代: 10轮 x 3次运行")
    
    # 计算最终HPWL
    final_hpwl = optimizer._calculate_current_hpwl(optimized_modules)
    initial_hpwl = optimizer._calculate_current_hpwl(original_modules)
    
    # 防止除零错误
    if initial_hpwl > 0:
        improvement = ((initial_hpwl - final_hpwl) / initial_hpwl) * 100
    else:
        improvement = 0.0
        print("\n警告: 初始HPWL为0，可能是连接数据异常")
    
    print(f"\nHPWL优化:")
    print(f"  - 初始HPWL: {initial_hpwl:.2f}")
    print(f"  - 最终HPWL: {final_hpwl:.2f}")
    print(f"  - 改善幅度: {improvement:.2f}%")
    
    # ========================================
    # 阶段2：DRC间距规则检查演示
    # ========================================
    print("\n" + "=" * 70)
    print("  【阶段2功能演示】DRC间距规则检查")
    print("=" * 70)
    
    # 对优化后的布局执行DRC检查
    drc_summary = optimizer.print_drc_report(optimized_modules, verbose=True)
    # 生成DRC违规可视化
    optimizer.visualize_drc_violations(optimized_modules, drc_summary, filename="drc_violations.png")
    
    print("\n布局优化系统运行完成！")


if __name__ == "__main__":
    main()
