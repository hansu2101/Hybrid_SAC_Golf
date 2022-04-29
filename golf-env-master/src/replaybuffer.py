import numpy as np
from collections import deque
import random

class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    ## 버퍼에 저장
    def add_buffer(self, state_img, state_dist, action_d, action_c, reward, next_state_img, next_state_dist, done):
        transition = (state_img, state_dist, action_d, action_c, reward, next_state_img, next_state_dist, done)

        # 버퍼가 꽉 찼는지 확인
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: # 찼으면 가장 오래된 데이터 삭제하고 저장
            self.buffer.popleft()
            self.buffer.append(transition)

    ## 버퍼에서 데이터 무작위로 추출 (배치 샘플링)
    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        # 상태, 행동, 보상, 다음 상태별로 정리
        states_img = np.asarray([i[0] for i in batch])      # state
        states_dist = np.asarray([i[1] for i in batch])
        actions_d = np.asarray([i[2] for i in batch])
        actions_c = np.asarray([i[3] for i in batch])
        rewards = np.asarray([i[4] for i in batch])
        next_states_img = np.asarray([i[5] for i in batch])
        next_states_dist = np.asarray([i[6] for i in batch])
        dones = np.asarray([i[7] for i in batch])

        return states_img, states_dist, actions_d, actions_c, rewards, next_states_img, next_states_dist, dones


    ## 버퍼 사이즈 계산
    def buffer_count(self):
        return self.count

    ## 버퍼 비움
    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0