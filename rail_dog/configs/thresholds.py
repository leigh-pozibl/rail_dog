from typing import Optional, Set, List

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
                

@dataclass
class TQIThresholds:
    status: str
    min: int
    max: int


class TQI_THRESHOLDS():
    def __init__(self):
        self.good = TQIThresholds("good", 0, 15)
        self.satisfactory = TQIThresholds("satisfactory", 15, 20)
        self.poor = TQIThresholds("poor", 20, 30)
        self.critical = TQIThresholds("critical", 30, 999)


    def get_status(self, avg_value: float) -> str:
        for attr in dir(self):
            threshold = getattr(self, attr)
            if isinstance(threshold, TQIThresholds):
                if threshold.min <= avg_value < threshold.max:
                    return threshold.status
        return "unknown"


PLAINLINE_TREATMENT_THRESHOLDS = {
    "gbfi_bog_threshold_1": 40,                 # T8 - gbfi_max_avg threshold for bog hole
    "gbfi_bog_threshold_2": 40,                 # T9 - gbfi_max_avg threshold for bog hole (alternative)
    "ballast_clean_threshold": 20,              # V10 - gbfi_avg threshold for clean (greater than)
    "lift_tamp_gbfi_avg_threshold": 20,         # V11 - gbfi_avg threshold for lift & tamp (less than)
    "lift_tamp_ballast_depth_threshold": 0.25,  # Z11 - ballast_depth threshold for lift & tamp (less than)
    "tamp_only_gbfi_avg_threshold": 20,         # V12 - gbfi_avg threshold for tamp only (less than)
    "tamp_only_ballast_depth_threshold": 0.25,  # Z12 - ballast_depth threshold for tamp only (greater than)
}


LX_TREATMENT_THRESHOLDS = {
    "lx_renewal_gbfi_max_threshold": 40,        # T14 - gbfi_max_avg threshold for Level Crossing Renewal
    "lx_renewal_gbfi_avg_threshold": 30,        # V14 - gbfi_avg_avg threshold for Level Crossing Renewal

    "tamp_only_gbfi_avg_threshold": 30,         # V18 - gbfi_avg threshold for tamp only (greater than)
    "tamp_only_ballast_depth_threshold": 0.25,  # Z12 - ballast_depth threshold for lift & tamp (greater than)

    "gbfi_bog_threshold_2": 40,                 # T9 - gbfi_max_avg threshold for bog hole (alternative)
    "ballast_clean_threshold": 20,              # V10 - gbfi_avg threshold for clean (greater than)
    "lift_tamp_gbfi_avg_threshold": 20,         # V11 - gbfi_avg threshold for lift & tamp (less than)
    "lift_tamp_ballast_depth_threshold": 0.25,  # Z11 - ballast_depth threshold for lift & tamp (less than)

}


IRJ_TREATMENT_THRESHOLDS = {
    "irj_renewal_gbfi_max_threshold": 40,       # T15 - gbfi_max_avg threshold for IRJ Renewal
    "tamp_only_gbfi_avg_threshold": 30,         # V18 - gbfi_avg threshold for tamp only (greater than)
    "tamp_only_ballast_depth_threshold": 0.25,  # Z18 - ballast_depth threshold for tamp only (greater than or equal)
}


TURNOUT_TREATMENT_THRESHOLDS = {
    "turnout_renewal_ballast_depth_threshold": 0.25,  # Z16 - ballast_depth threshold for turnout renewal
    "tamp_only_gbfi_avg_threshold": 30,               # V18 - gbfi_avg threshold for tamp only (greater than)
    "tamp_only_ballast_depth_threshold": 0.25,        # Z18 - ballast_depth threshold for tamp only (greater than or equal)
}


BRIDGE_TREATMENT_THRESHOLDS = {
    "bridge_renewal_ballast_depth_threshold": 0.25,   # Z17 - ballast_depth threshold for bridge renewal
    "bridge_renewal_gbfi_avg_threshold": 20,          # V17 - gbfi_avg threshold for bridge renewal (greater than)
    "tamp_only_gbfi_avg_threshold": 30,               # V18 - gbfi_avg threshold for tamp only (greater than)
    "tamp_only_ballast_depth_threshold": 0.25,        # Z18 - ballast_depth threshold for tamp only (greater than or equal)
}

PRIORITY_THRESHOLDS = {
    # Weights (sum to 1.0)
    "priority_weight_tsr": 0.25,
    "priority_weight_dtf": 0.25,
    "priority_weight_tqi_status": 0.15,
    "priority_weight_tqi_trend": 0.15,
    "priority_weight_speed": 0.15,
    "priority_weight_curve": 0.05,
    # Normalization values (max possible score for each factor)
    "priority_norm_tsr": 1,
    "priority_norm_dtf": 2,
    "priority_norm_tqi_status": 2,
    "priority_norm_tqi_trend": 3,
    "priority_norm_speed": 2,
    "priority_norm_curve": 2,
    # Speed thresholds
    "priority_speed_low": 45,
    "priority_speed_high": 65,
    "priority_speed_min": 0,
    "priority_speed_mid": 46,
    # LX priority weights
    "priority_lx_weight_fixed_asset": 0.3,
    "priority_lx_weight_tsr": 0.25,
    "priority_lx_weight_dtf": 0.2,
    "priority_lx_weight_dtf_worse": 0.10,
    "priority_lx_weight_speed": 0.15,
    # LX priority normalization
    "priority_lx_norm_fixed_asset": 2,
    "priority_lx_norm_tsr": 1,
    "priority_lx_norm_dtf": 2,
    "priority_lx_norm_dtf_worse": 1,
    "priority_lx_norm_speed": 0.15,
    
}

# @dataclass
# class TreatmentThresholds:
#     treatment:str
#     tqi_status: Optional[Set[str]] = None
#     gbfi_max_avg: Optional[float] = None
#     gbfi_avg: Optional[float] = None
#     monash_ioc: Optional[Set[str]] = None
#     tsr_history: Optional[bool] = None
#     ballast_depth: Optional[float] = None
#     wo_thresholds_exceedance: Optional[bool] = None

# class PLAINLINE_TREATMENT_THRESHOLDS():
#     def __init__(self):
#         self.bog_hole_1 = TreatmentThresholds(
#             treatment="Local Formation Renewal (Bog Hole)",
#             tqi_status=set("poor", "critical"),
#             gbfi_max_avg=40,
#         )
#         self.bog_hole_2 = TreatmentThresholds(
#             treatment="Local Formation Renewal (Bog Hole)",
#             gbfi_max_avg=40,
#             monash_ioc=set("s1", "s2"),
#             tsr_history=True
#         )
#         self.ballast_clean = TreatmentThresholds(
#             treatment="Ballast Clean",
#             tqi_status=set("poor", "critical"),
#             gbfi_avg=20,
#         )
#         self.lift_tamp = TreatmentThresholds(
#             treatment="Lift & Tamp",
#             tqi_status=set("poor", "critical"),
#             gbfi_avg=20,
#             ballast_depth=0.25
#         )
#         self.tamp_only = TreatmentThresholds(
#             treatment="Tamp Only",
#             tqi_status=set("poor", "critical"),
#             gbfi_avg=20,
#             ballast_depth=0.25
#         )

        										