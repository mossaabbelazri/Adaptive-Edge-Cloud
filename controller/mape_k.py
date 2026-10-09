import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAPEK-Controller")

class MAPEKLoop:
    """
    MAPE-K (Monitor, Analyze, Plan, Execute, Knowledge) Autonomic Control Loop
    (Kephart & Chess, IBM 2003) for Selective Inference Cascades (Geifman & El-Yaniv, 2017).
    
    Theoretical Formulation:
    ------------------------
    Balances the trade-off between Edge compute autonomy and Cloud ingestion capacity.
    Formulated as a dynamic constrained optimization problem:
        min_{theta} Ingestion_Bandwidth(theta)
        subject to: Selective_Risk(f, g_theta) <= epsilon_target
                    Cloud_Capacity_Ratio <= C_max (e.g., 30%)
    
    The threshold theta acts as a dynamic Lagrangian multiplier balancing coverage
    and communication overhead under non-stationary physical distribution shift.
    """
    def __init__(self, initial_threshold=0.8):
        self.threshold = initial_threshold
        self.knowledge_base = {
            'uncertain_ratio_history': [],
            'max_cloud_capacity_ratio': 0.3 # SLA constraint: Max 30% of traffic offloaded
        }

    def monitor(self, total_samples, uncertain_samples):
        """
        [Monitor]: Collects empirical offload ratio Phi(g) = E[g(X)] over sliding window.
        """
        ratio = uncertain_samples / total_samples if total_samples > 0 else 0.0
        self.knowledge_base['uncertain_ratio_history'].append(ratio)
        if len(self.knowledge_base['uncertain_ratio_history']) > 10:
            self.knowledge_base['uncertain_ratio_history'].pop(0)
        return ratio

    def analyze(self, current_ratio):
        """
        [Analyze]: Determines if the system violates SLA / Capacity bounds C_max.
        """
        avg_ratio = sum(self.knowledge_base['uncertain_ratio_history']) / len(self.knowledge_base['uncertain_ratio_history'])
        
        # If offloading exceeds SLA budget, tighten rejection criteria (raise threshold)
        if avg_ratio > self.knowledge_base['max_cloud_capacity_ratio']:
            return "increase_threshold"
        # If cloud bandwidth is underutilized, relax threshold to lower Selective Risk on Edge
        elif avg_ratio < (self.knowledge_base['max_cloud_capacity_ratio'] / 2):
            return "decrease_threshold"
        return "maintain"

    def plan(self, analysis_result):
        """
        [Plan]: Computes the gradient update on the decision boundary threshold theta.
        """
        step = 0.05
        if analysis_result == "increase_threshold":
            return min(2.0, self.threshold + step)
        elif analysis_result == "decrease_threshold":
            return max(0.1, self.threshold - step)
        return self.threshold

    def execute(self, new_threshold):
        """
        [Execute]: Actuates the updated threshold in the streaming runtime.
        """
        if new_threshold != self.threshold:
            logger.info(f"[Adaptation] Threshold adjusted from {self.threshold:.2f} to {new_threshold:.2f} (Selective Risk vs Ingestion balancing)")
            self.threshold = new_threshold

    def step(self, total_samples, uncertain_samples):
        """
        Executes a single discrete-time iteration of the closed-loop controller.
        """
        ratio = self.monitor(total_samples, uncertain_samples)
        analysis = self.analyze(ratio)
        new_thresh = self.plan(analysis)
        self.execute(new_thresh)
        return self.threshold

