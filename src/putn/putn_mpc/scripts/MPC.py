import numpy as np
import casadi as ca
import time


def MPC(self_state, goal_state, obstacles):
    opti = ca.Opti()
    ## parameters for optimization
    T = 0.2
    N = 10  # MPC horizon
    v_max = 0.5
    omega_max = 0.6
    safe_distance = 0.55
    Q = np.array([[1.2, 0.0, 0.0],[0.0, 1.2, 0.0],[0.0, 0.0, 0.0]])
    R = np.array([[0.2, 0.0], [0.0, 0.15]])
    goal = goal_state[:,:3]
    tra = goal_state[:,2]
    conf = goal_state[:,3]
    tra_mean = np.mean(tra)
    opt_x0 = opti.parameter(3)
    opt_controls = opti.variable(N, 2)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]

    ## state variables
    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    ## create funciont for F(x)
    theta_x = self_state[0][4]*np.cos(self_state[0][2]) - self_state[0][3]*np.sin(self_state[0][2])
    theta_y = self_state[0][4]*np.sin(self_state[0][2]) + self_state[0][3]*np.cos(self_state[0][2])
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2])*np.cos(theta_x), u_[0]*ca.sin(x_[2])*np.cos(theta_y), u_[1]])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    
    # Acceleration constraints (for smoother motion)
    acc_v_max = 0.2  # max linear acceleration (m/s^2)
    acc_w_max = 0.5  # max angular acceleration (rad/s^2)
    
    # 初始控制输入 (上一时刻的控制输入)，用于计算第一步的加速度
    # 这里简化处理，假设初始控制为0，或者可以作为参数传入
    # 为了简化，我们只约束 MPC 预测步骤之间的加速度
    
    for i in range(N-1):
        # Linear acceleration constraint: |v[i+1] - v[i]| / T <= acc_v_max
        opti.subject_to(opti.bounded(-acc_v_max * T, v[i+1] - v[i], acc_v_max * T))
        
        # Angular acceleration constraint: |w[i+1] - w[i]| / T <= acc_w_max
        opti.subject_to(opti.bounded(-acc_w_max * T, omega[i+1] - omega[i], acc_w_max * T))

    # Position Boundaries
    # Here you can customize the avoidance of local obstacles 

    # Admissable Control constraints
    # 禁止后向速度：设置最小速度为 0
    # 注意：如果只是将v_min设为0，当目标在后方时，可能会导致无法产生旋转，
    # 因此需要在cost function中增加对朝向误差的惩罚，促使车辆原地旋转。
    
    # 允许倒车的标志位，可以通过参数传入，这里为了满足需求设为False
    allow_reverse = False
    
    if allow_reverse:
        opti.subject_to(opti.bounded(-v_max, v, v_max))
    else:
        # 禁止倒车，速度下限设为0
        opti.subject_to(opti.bounded(0, v, v_max))
        
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max)) 

    # System Model constraints
    for i in range(N):
        x_next = opt_states[i, :] + T*f(opt_states[i, :], opt_controls[i, :]).T
        opti.subject_to(opt_states[i+1, :]==x_next)

    #### cost function
    obj = 0 
    
    # 增加 heading alignment cost 的权重
    # 计算当前位置到目标的朝向角
    # 我们使用最后一个目标点来计算大致的期望朝向
    
    # 增加朝向对齐的惩罚项权重，促使车辆在原地旋转以对准目标
    # 如果禁止倒车，需要大幅增加这个权重
    theta_align_weight = 3.0 if not allow_reverse else 0.5
    
    # 增加对前进速度的奖励（鼓励在朝向正确时向前移动），防止在原地过度犹豫
    progress_weight = 3.0  # 进一步增加前进奖励权重
    
    # 增加对旋转速度的惩罚（平滑旋转，抑制振荡）
    omega_damp_weight = 0.5 # 增加阻尼，减少快速来回摆动
    
    # 对齐误差的阈值，超过此阈值时强烈鼓励旋转
    theta_tol_hi = 1.5 # 约30度
    theta_tol_lo = 1.0 # 放宽对齐要求，约23度，只要大致对准就开始前进
    
    # 添加一个偏置，鼓励优先向目标方向旋转（sign(heading_error) * omega > 0）
    omega_bias_weight = 0.5 # 增加偏置权重，明确旋转方向
    
    # 增加一个最小前向速度约束（当对齐较好时），强制车辆移动
    v_track_weight = 0.5

    # 预瞄步数，用于计算 heading_to_goal，避免对当前点的过度敏感
    lookahead_steps = 3

    for i in range(N):
        # 计算当前步到目标的期望朝向 (简单的朝向目标点)
        # 使用预瞄点的方向，使路径更平滑，减少因跟踪最近点导致的振荡
        j = min(N-1, i + lookahead_steps)
        
        # 获取预瞄步骤对应的目标点
        current_goal = goal[j]
        
        dx = current_goal[0] - opt_states[i, 0]
        dy = current_goal[1] - opt_states[i, 1]
        
        # 计算期望朝向
        heading_to_goal = ca.atan2(dy, dx)
        
        # 计算朝向误差 (考虑角度循环性)
        heading_error = opt_states[i, 2] - heading_to_goal
        # 归一化到 [-pi, pi]
        heading_error = ca.atan2(ca.sin(heading_error), ca.cos(heading_error))
        
        # 使用 1 - cos(error) 作为对齐代价，范围 [0, 2]
        align_cost = 1.0 - ca.cos(heading_error)
        
        # 基础的状态误差和控制量代价
        state_cost = 0.1 * ca.mtimes([(opt_states[i, :] - goal[[i]]), Q, (opt_states[i, :]- goal[[i]]).T])
        control_cost = ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
        
        # 额外的旋转平滑代价 - 惩罚角速度的变化率（这里简化为惩罚角速度本身，防止过大）
        # 更理想的是惩罚 d(omega)/dt，但这里只有控制量，可以惩罚 omega^2
        omega_cost = omega_damp_weight * (opt_controls[i, 1]**2)
        
        # 旋转方向偏置：强迫旋转方向与误差消除方向一致
        # 增加一个死区，当误差很小时不强制，避免微小误差导致的抖动
        heading_sign = ca.tanh(heading_error / 0.1) # 变陡，对小误差也敏感，但在0附近平滑
        # 只有当 omega 的方向与消除误差的方向相反时才惩罚
        # 期望 omega * heading_error < 0 (即 sign(omega) == -sign(error))
        # 惩罚项： max(0, -omega * -sign(error)) = max(0, omega * sign(error)) -- 不对，应该是反过来
        # 误差>0 (车头偏左)，需要右转 (omega<0)。此时 error*omega < 0。
        # 如果 omega>0 (左转)，则 error*omega > 0，需要惩罚。
        # 惩罚项 = max(0, omega * heading_sign)
        rotate_gain = ca.fmax(0.0, ca.fabs(heading_error) - 0.05) # 死区 0.05 rad
        omega_sign_cost = omega_bias_weight * rotate_gain * ca.fmax(0, opt_controls[i, 1] * heading_sign)
        
        # 前进奖励：只有当朝向对齐较好时（align_cost 小），才鼓励前进速度
        # 放宽对齐条件，使用高斯函数形式 exp(-error^2) 作为权重
        alignment_factor = ca.exp(- (heading_error**2) / (theta_tol_lo**2)) 
        progress_reward = - progress_weight * alignment_factor * opt_controls[i, 0]
        
        # 跟踪速度奖励：直接鼓励速度接近参考速度（如果有）或者只是鼓励动起来
        # 这里简单鼓励 v > 0
        v_track_cost = - v_track_weight * opt_controls[i, 0]

        # 总代价包含朝向对齐代价
        obj = obj + state_cost + control_cost + theta_align_weight * align_cost + omega_cost + progress_reward + omega_sign_cost + v_track_cost
        
        # 如果禁止倒车，添加一个软约束：当朝向误差很大时，线速度应该很小
        if not allow_reverse:
             # 当 heading_error^2 > theta_tol_hi^2 时，施加 v 的惩罚
             # 使用平滑的惩罚函数
             misalign_factor = 1.0 - ca.exp(- (heading_error**2) / (theta_tol_hi**2))
             v_penalty = 10.0 * misalign_factor * (opt_controls[i, 0]**2)
             obj = obj + v_penalty
             
             # 额外：当朝向误差较小 (例如 < 0.2 rad) 时，添加最小速度约束，防止静止
             # 注意：这是硬约束，可能导致不可行，所以用软约束或者在求解器外处理
             # 这里我们尝试在 cost 中加入对 v 过小的惩罚（即鼓励 v 接近 v_max 或某个期望值）
             # 上面的 v_track_cost 已经起到了这个作用

    # 终端代价
    obj = obj + 2*ca.mtimes([(opt_states[N-1, :] - goal[[N-1]]), Q, (opt_states[N-1, :]- goal[[N-1]]).T])

    opti.minimize(obj)
    opts_setting = {'ipopt.max_iter':80, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-3, 'ipopt.acceptable_obj_change_tol':1e-3}
    opti.solver('ipopt',opts_setting)
    opti.set_value(opt_x0, self_state[:,:3])

    try:
        sol = opti.solve()
        u_res = sol.value(opt_controls)
        state_res = sol.value(opt_states)
    except:
        state_res = np.repeat(self_state[:3],N+1,axis=0)
        u_res = np.zeros([N,2])

    return state_res, u_res
