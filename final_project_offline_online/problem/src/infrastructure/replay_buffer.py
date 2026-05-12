from infrastructure.utils import *


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None

    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }

    def sample_chunk(self, batch_size: int, chunk_size: int, discount: float = 0.99):
        """Sample K-step action chunks by aggregating consecutive transitions."""
        if chunk_size == 1:
            return self.sample(batch_size)

        max_valid = self.size - chunk_size
        if max_valid <= 0:
            return self.sample(batch_size)

        rand_starts = np.random.randint(0, max_valid, size=(batch_size,))
        start_idx = rand_starts % self.max_size

        obs = self.observations[start_idx].copy()
        action_chunks = []
        cum_rewards = np.zeros(batch_size, dtype=np.float32)
        alive = np.ones(batch_size, dtype=np.float32)
        final_next_obs = self.next_observations[start_idx].copy()
        final_dones = np.zeros(batch_size, dtype=np.float32)

        for k in range(chunk_size):
            idx = (rand_starts + k) % self.max_size
            action_chunks.append(self.actions[idx].copy())
            cum_rewards += alive * (discount ** k) * self.rewards[idx]
            alive_mask = alive > 0
            final_next_obs[alive_mask] = self.next_observations[idx[alive_mask]]
            done_now = self.dones[idx].astype(np.float32)
            final_dones = np.maximum(final_dones, alive * done_now)
            alive = alive * (1.0 - done_now)

        return {
            "observations": obs,
            "actions": np.concatenate(action_chunks, axis=-1),
            "rewards": cum_rewards,
            "next_observations": final_next_obs,
            "dones": final_dones,
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observation.shape), dtype=observation.dtype
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_observation.shape), dtype=next_observation.dtype
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.next_observations.shape[1:]
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done

        self.size += 1
