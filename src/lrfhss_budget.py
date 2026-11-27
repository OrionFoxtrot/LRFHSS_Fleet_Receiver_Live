"""Optional bounded pressure policy. Thresholds informed by prior backlog (85/256)."""
class AdaptiveHeaderBudget:
    LOWER=2
    UPPER=4
    def __init__(self):
        self.active=2;self.high_streak=0;self.low_streak=0;self.transitions=0
    def update(self,backlog,oldest_age_s):
        high=backlog>=64 or (backlog>0 and oldest_age_s>=.75)
        low=backlog<=16 and oldest_age_s<.35
        self.high_streak=self.high_streak+1 if high else 0
        self.low_streak=self.low_streak+1 if low else 0
        before=self.active
        if self.active==2 and self.high_streak>=2:self.active=4
        elif self.active==4 and self.low_streak>=8:self.active=2
        self.transitions+=int(before!=self.active)
        return self.active
