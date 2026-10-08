import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAPEK-Controller")

class MAPEKLoop:
    """
    MAPE-K (Monitor, Analyze, Plan, Execute, Knowledge) self-adaptation loop.
    Controls the dynamic offloading threshold based on system state.
    """
    def __init__(self, initial_threshold=0.8):
        self.threshold = initial_threshold
        self.knowledge_base = {
            'uncertain_ratio_history': [],
            'max_cloud_capacity_ratio': 0.3 # Max 30% of traffic can go to cloud
        }

    def monitor(self, total_samples, uncertain_samples):
        """ Collects metrics from the managed system. """
        ratio = uncertain_samples / total_samples if total_samples > 0 else 0.0
        self.knowledge_base['uncertain_ratio_history'].append(ratio)
        if len(self.knowledge_base['uncertain_ratio_history']) > 10:
            self.knowledge_base['uncertain_ratio_history'].pop(0)
        return ratio

    def analyze(self, current_ratio):
        """ Analyzes the metrics to determine if adaptation is needed. """
        avg_ratio = sum(self.knowledge_base['uncertain_ratio_history']) / len(self.knowledge_base['uncertain_ratio_history'])
        
        # If we are sending too much to the cloud, we need to increase the threshold (be less sensitive)
        if avg_ratio > self.knowledge_base['max_cloud_capacity_ratio']:
            return "increase_threshold"
        # If we have spare cloud capacity and the recent ratio is low, we can lower the threshold (be more sensitive)
        elif avg_ratio < (self.knowledge_base['max_cloud_capacity_ratio'] / 2):
            return "decrease_threshold"
        return "maintain"

    def plan(self, analysis_result):
        """ Plans the adaptation steps based on the analysis. """
        step = 0.05
        if analysis_result == "increase_threshold":
            return min(2.0, self.threshold + step)
        elif analysis_result == "decrease_threshold":
            return max(0.1, self.threshold - step)
        return self.threshold

    def execute(self, new_threshold):
        """ Executes the adaptation plan. """
        if new_threshold != self.threshold:
            logger.info(f"[Adaptation] Threshold adjusted from {self.threshold:.2f} to {new_threshold:.2f}")
            self.threshold = new_threshold

    def step(self, total_samples, uncertain_samples):
        """ Runs a full MAPE-K cycle. """
        ratio = self.monitor(total_samples, uncertain_samples)
        analysis = self.analyze(ratio)
        new_thresh = self.plan(analysis)
        self.execute(new_thresh)
        return self.threshold
