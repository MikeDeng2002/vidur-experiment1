from typing import List, Tuple
import numpy as np
import logging
import os
from scheduler_predictor import SchedulerPredictor
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from vidur.scheduler.global_scheduler.lor3_global_scheduler import LOR3GlobalScheduler
from vidur.scheduler.global_scheduler.lor4_global_scheduler import LOR4GlobalScheduler
from vidur.scheduler.global_scheduler.lor5_global_scheduler import LOR5GlobalScheduler

class OnlineAdaptiveGlobalScheduler(BaseGlobalScheduler):
    """
    Adaptive global scheduler: every time 100 completed requests are collected,
    compute window features and use SchedulerPredictor to predict the best scheduler (lor/lor3/lor4/lor5),
    and dynamically switch scheduler for schedule().
    """
    def __init__(self, config, replicas):
        super().__init__(config, replicas)
        self.completed_requests = []
        self.window_size = 100
        
        # Try to load pre-trained model, if not available, train a new one
        model_path = 'trained_scheduler_predictor.joblib'
        if os.path.exists(model_path):
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Loading pre-trained model from {model_path}")
            self.predictor = SchedulerPredictor(model_path=model_path)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.info("No pre-trained model found, training new model...")
            self.predictor = SchedulerPredictor()
            # Save the trained model for future use
            self.predictor.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
        
        self.current_scheduler_name = 'lor'
        self.schedulers = {
            'lor': LORGlobalScheduler(config, replicas),
            'lor3': LOR3GlobalScheduler(config, replicas),
            'lor4': LOR4GlobalScheduler(config, replicas),
            'lor5': LOR5GlobalScheduler(config, replicas)
        }
        self.current_scheduler = self.schedulers[self.current_scheduler_name]

    def on_request_completed(self, request):
        """
        Call this method whenever a request is completed.
        Maintains a sliding window of completed requests, and updates the scheduler if needed.
        """
        self.completed_requests.append(request)
        if len(self.completed_requests) > self.window_size:
            self.completed_requests.pop(0)
        if len(self.completed_requests) == self.window_size:
            features = self.compute_features(self.completed_requests)
            pred = self.predictor.predict(features)
            # Use the main metric, e.g., best_TFT_P95
            best_sched = pred['best_TFT_P95']
            if best_sched != self.current_scheduler_name:
                self.logger.info(f"Switching scheduler from {self.current_scheduler_name} to {best_sched}")
                self.current_scheduler_name = best_sched
                self.current_scheduler = self.schedulers[best_sched]

    def compute_features(self, requests):
        """
        Compute features from a list of completed requests for prediction.
        """
        timestamps = [r._arrived_at for r in requests]
        inter_arrival_times = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        mean_inter_arrival_time = np.mean(inter_arrival_times) if inter_arrival_times else 0
        burstiness = np.std(inter_arrival_times) / mean_inter_arrival_time if mean_inter_arrival_time > 0 else 0
        prefill_tokens = [getattr(r, 'num_prefill_tokens', 0) for r in requests]
        mean_prefill_tokens = np.mean(prefill_tokens)
        std_prefill_tokens = np.std(prefill_tokens)
        decode_tokens = [getattr(r, 'num_decode_tokens', 0) for r in requests]
        mean_decode_tokens = np.mean(decode_tokens)
        std_decode_tokens = np.std(decode_tokens)
        return {
            'inter_arrival_time': mean_inter_arrival_time,
            'burstiness': burstiness,
            'mean_prefill': mean_prefill_tokens,
            'std_prefill': std_prefill_tokens,
            'mean_decode': mean_decode_tokens,
            'std_decode': std_decode_tokens
        }

    def schedule(self) -> List[Tuple[int, object]]:
        """
        Delegate scheduling to the currently selected scheduler.
        Ensure all requests in queue are scheduled.
        """
        # First, ensure request queue is sorted
        self.sort_requests()
        
        # Copy current request queue to the current scheduler
        self.current_scheduler._request_queue = self._request_queue.copy()
        
        # Get assignments from current scheduler
        assignments = self.current_scheduler.schedule()
        
        # Clear our request queue since requests have been assigned
        self._request_queue.clear()
        
        # Log scheduling info
        if assignments:
            self.logger.debug(f"Scheduled {len(assignments)} requests using {self.current_scheduler_name}")
        
        return assignments

    def is_empty(self) -> bool:
        """
        Check if both the main queue and all replica schedulers are empty.
        """
        return (len(self._request_queue) == 0 and 
                all(scheduler.is_empty() for scheduler in self.schedulers.values()))