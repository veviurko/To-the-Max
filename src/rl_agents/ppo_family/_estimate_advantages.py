import torch


def _estimate_advantages_gae(value, reward, done, trunc, obs_next, gae_lambda, device, evaluate_v, discount):
    rollout_length = value.shape[1]
    advantages = torch.zeros_like(reward).to(device)
    lastgaelam = 0
    with torch.no_grad():
        values_last_t = evaluate_v(obs_next[:, -1]).reshape(-1, 1)
        for t in reversed(range(rollout_length)):
            if t == rollout_length - 1:
                next_values = values_last_t
            else:
                next_values = value[:, t + 1]

            delta = reward[:, t] + ~done[:, t] * discount * next_values - value[:, t]
            advantages[:, t] = lastgaelam = delta + discount * gae_lambda * ~done[:, t] * lastgaelam
    return advantages


def _estimate_advantages_montecarlo(value, reward, done, trunc, obs_next, device, evaluate_v, discount):
    rollout_length = value.shape[1]
    returns = torch.zeros_like(reward).to(device)
    with torch.no_grad():
        values_last_t = evaluate_v(obs_next[:, -1]).reshape(-1, 1)
        for t in reversed(range(rollout_length)):
            if t == rollout_length - 1:
                next_return = values_last_t
            else:
                next_return = returns[:, t + 1]
            returns[:, t] = reward[:, t] + discount * ~done[:, t] * next_return
        advantages = returns - value
    return advantages


def _estimate_maxdet_advantages_montecarlo(value, reward, done, trunc, obs_next, device, evaluate_v, discount):
    rollout_length = value.shape[1]
    returns = torch.zeros_like(reward).to(device)
    with torch.no_grad():
        values_last_t = evaluate_v(obs_next[:, -1]).reshape(-1, 1)
        for t in reversed(range(rollout_length)):
            if t == rollout_length - 1:
                next_return = values_last_t
            else:
                next_return = returns[:, t + 1]
            returns[:, t] = torch.maximum(reward[:, t], discount * ~done[:, t] * next_return)
        advantages = returns - value
    return advantages


def _estimate_max_returns_montecarlo(y, reward, done, trunc, obs_next, device, evaluate_v, discount):
    rollout_length = reward.shape[1]
    with (torch.no_grad()):
        last_obs = torch.nan * torch.ones_like(obs_next)        # The last observation at the truncation moment
        factor = torch.ones_like(reward)                        # Discount until last_obs
        returns = torch.zeros_like(reward).to(device)           # Bootstrapped MC estimate of the return
        in_finished_traj = torch.zeros_like(done).to(device)    # If s[t] belongs to a trajectory that is finished
        traj_max_after_t = torch.zeros_like(reward).to(device)  # Observed max discounted reward after s[t]
        factor[:, -1] = discount
        last_obs[:, -1] = obs_next[:, -1]
        in_finished_traj[:, -1] = done[:, -1]
        traj_max_after_t[:, -1] = reward[:, -1]
        y_next = torch.max(y[:, -1], reward[:, -1])
        returns[:, -1] = torch.where(done[:, -1], y_next, discount * evaluate_v(obs_next[:, -1], y_next / discount))
        returns[:, -1] = torch.max(y_next, returns[:, -1])
        # print('t=', rollout_length - 1, 'obs:', obs_next[:, -1].abs().sum(), 'y', y_next, 'factor=', discount)
        # print('result', returns[:, -1])
        for t in reversed(range(0, rollout_length - 1)):
            # For truncated trajectories, last obs becomes 'obs_next' at the moment of truncation
            # We don't care about trajectories that terminate with "done" here, as they will not be bootstrapped
            last_obs[:, t] = last_obs[:, t + 1]
            last_obs[:, t][trunc[:, t, 0]] = obs_next[:, t][trunc[:, t, 0]]
            # For trajectories that are not truncated, we multiply the factor by 'discount'.
            factor[:, t] = discount * factor[:, t + 1]
            factor[:, t][trunc[:, t, 0]] = discount
            # Check whether trajectory is done. We needed for bootstrapping.
            in_finished_traj[:, t] = in_finished_traj[:, t] | done[:, t]
            # Compute the discounted max reward from current moment until done/truncation moment.
            r_t = reward[:, t]
            r_max_bootstrap = torch.where(done[:, t] | trunc[:, t], -torch.inf, traj_max_after_t[:, t + 1])
            traj_max_after_t[:, t] = torch.max(r_t, discount * r_max_bootstrap)
            y_target = torch.max(traj_max_after_t[:, t], y[:, t])
            returns[:, t] = torch.where(in_finished_traj[:, t], y_target,
                                        factor[:, t] * evaluate_v(last_obs[:, t], y_target / factor[:, t]))
            # print('t=', t, 'obs:', last_obs[:, t].abs().sum(), 'y', y_target, 'factor', factor[:, t])
            # print('result', returns[:, t])
            returns[:, t] = torch.max(y_target, returns[:, t])

    # print('Returns:', returns[:, :, 0])
    return returns


def _estimate_max_returns_gae(y, reward, done, trunc, obs_next, device, evaluate_v, discount, gae_lambda=0.95):
    assert not torch.any(done), 'Only truncated trajectories are supported in this implementation of max-GAE'
    rollout_length = reward.shape[1]
    returns = torch.zeros_like(reward)
    # print('returns in max-gae:', returns.shape)
    # It is important trajectories from different batches align.
    with (torch.no_grad()):
        # R_t_n_matrix = torch.nan * torch.ones(rollout_length, rollout_length, 1).to(device)
        for t in range(rollout_length):
            # We compute returns from the state s_t
            factor = float(discount)
            lambda_factor = 1.0
            for n in range(0, rollout_length - t):
                # We compute all n-returns until trajectory is truncated
                y_next = torch.maximum(y[:, t + n], reward[:, t + n])
                v_t_n = factor * evaluate_v(obs_next[:, t + n], y_next / factor)
                # print('v_t_n', v_t_n.shape)

                if torch.any(trunc[:, t + n]) or t + n == rollout_length - 1:
                    # print('t=', t, 'obs:', obs_next[:, t + n].abs().sum(), 'y_next', y_next, 'factor', factor)
                    returns[:, t] += lambda_factor * v_t_n
                    # print('result', returns[:, t])
                    break
                else:
                    returns[:, t] += (1-gae_lambda) * lambda_factor * v_t_n
                    factor *= discount
                    lambda_factor *= gae_lambda
            returns[:, t] = torch.max(y_next, returns[:, t])

    # print('Returns:', returns[:, :, 0])
    return returns
