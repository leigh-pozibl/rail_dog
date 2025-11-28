import pandas as pd

from pydantic.dataclasses import dataclass


class RAIL_DEGRADATION_THRESHOLDS():
    """
    Rail degradation thresholds.
    note: lo rail is the inside rail of a curve, hi rail is the outside rail
        A RH turn (as perceived when travelling away from the port) is a turn to the west -> the lo rail is the west rail
        A LH turn (as perceived when travelling away from the port) is a turn to the east -> the lo rail is the east rail
    """
    def __init__(self):
        self.tangent = {
            "vert_wear": 16,
            "side_wear": 4,
            "rel_head_loss": {"lo": 40, "hi": 40},
        }
        self.mild_curve = {
            "vert_wear": 14,
            "side_wear": 6,
            "rel_head_loss": {"lo": 32, "hi": 37},
        }
        self.sharp_curve = {
            "vert_wear": 10,
            "side_wear": 6,
            "rel_head_loss": {"lo": 25, "hi": 25},
        }

    def get(self, clas, param, hand=None, track=None):
        d = getattr(self, clas)
        if param != "rel_head_loss":
            if param in d:
                return d[param]
            else:
                raise ValueError(f"Unrecognised param: {param} for class: {clas}")
        elif param == "rel_head_loss":
            rail = self._get_rail(hand, track)
            return d[param][rail]
        else:
            raise ValueError(f"Unrecognised param: {param} for class: {clas}")

    def _get_rail(self, hand, track):
        if hand == "RH":
            return "lo" if track == "w" else "hi"
        elif hand == "LH":
            return "lo" if track == "e" else "hi"
        else:
            if np.isnan(hand):
                # these are tangent sections, don't need to distinguish between left and right hand
                return "lo"
            else:
                raise ValueError(f"Unrecognised hand: {hand}")

         
@dataclass
class AssetWOThresholds:
    asset_type: str
    status: str
    min_count: int


class ASSET_WORK_ORDER_THRESHOLDS():
    def __init__(self):
        self.h_bridge = AssetWOThresholds("bridge", "h", 1)
        self.vh_bridge = AssetWOThresholds("bridge", "vh", 2)
        self.h_irj = AssetWOThresholds("irj", "h", 1)
        self.vh_irj = AssetWOThresholds("irj", "vh", 2)
        self.h_level_crossing = AssetWOThresholds("level_crossing", "h", 10)
        self.vh_level_crossing = AssetWOThresholds("level_crossing", "vh", 20)
        self.h_turnout = AssetWOThresholds("turnout", "h", 40)
        self.vh_turnout = AssetWOThresholds("turnout", "vh", 50)
        
        self.asset_types = {"bridge", "irj", "level_crossing", "turnout"}
        
        self.thresholds = {}
        for asset_type in self.asset_types:
            self.thresholds[asset_type] = [
                getattr(self, attr) for attr in dir(self)
                if isinstance(getattr(self, attr), AssetWOThresholds) and getattr(self, attr).asset_type == asset_type
            ]
        
    def get_status(self, asset_type: str, work_order_count: int) -> str:
        """Get the status for an asset type based on work order count"""
        if self.thresholds.get(asset_type) is None:
            return ""
        
        for threshold in sorted(self.thresholds[asset_type], key=lambda x: x.min_count, reverse=True):
            if work_order_count >= threshold.min_count:
                return threshold.status
        
        return ""
    
    def get_max_status(self, year_status: dict) -> str:
        """Get the maximum status from a dict of year_status values"""
        years_to_check = {year_status.get("2024"), year_status.get("2025")}
        
        if "vh" in years_to_check:
            return "vh"
        elif "h" in years_to_check:
            return "h"
        else: return ""


@dataclass
class GBFIFouledThresholds:
    status: str
    min: int
    max: int
    

class GBFI_FOULED_THRESHOLDS():
    def __init__(self):
        self.clean = GBFIFouledThresholds("clean", 0, 20)
        self.moderate = GBFIFouledThresholds("moderate", 10, 20)
        self.reasonable = GBFIFouledThresholds("reasonable", 20, 30)
        self.fouled = GBFIFouledThresholds("fouled", 30, 40)
        self.high = GBFIFouledThresholds("high", 40, 9999)

    def get_status(self, avg_value: float) -> str:
        for attr in dir(self):
            threshold = getattr(self, attr)
            if isinstance(threshold, GBFIFouledThresholds):
                if threshold.min <= avg_value < threshold.max:
                    return threshold.status
        return "unknown"
    
    def get_status_counts(self, data: pd.DataFrame) -> dict:
        status_counts = {}
        for attr in dir(self):
            threshold = getattr(self, attr)
            if isinstance(threshold, GBFIFouledThresholds):
                count = ((data["avg"] >= threshold.min) & (data["avg"] < threshold.max)).sum()
                setattr(self, f"{threshold.status}_count", count)
                status_counts[threshold.status] = count
        return status_counts
                
                
        