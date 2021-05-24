import os
import locale
import logging
import time
import datetime
import numpy as np
from python_settings import settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer

logger = logging.getLogger(__name__)

class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code # 종목 코드
        self.chart_data = chart_data
        self.environment = Environment(chart_data) # 환경 객체
        # 에이전트 객체
        self.agent = Agent(self.environment, min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit, delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data # 학습 데이터
        self.sample = None
        self.training_data_idx = -1
        # 정책 신경망; 입력 크기 = 학습 데이터의 크기 + 에이전트의 상태 크기
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTION, lr=lr)
        self.visualizer = Visualizer() # 가시화 모듈