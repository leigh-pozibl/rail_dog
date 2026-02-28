from dataclasses import dataclass
from typing import Optional
import logging

import pandas as pd


@dataclass
class Station:
    """Railway station with chainage boundaries."""
    name: str
    start_km: float
    end_km: float


@dataclass
class Section:
    """Railway section with network and chainage boundaries."""
    asset_name: str         # eg: "MLE_3.963_4.301"
    section_id: str         # eg: "ML Section 1"
    network: str            # eg: "Mainline1"
    start_km: float
    end_km: float
    line_code: str          # ML, TH, SL, EL
    line: str               # thomas, mainline, solomon, eliwana
    line_class: str         # main, bypass, loop

    
STATIONS = {
    "Port Loop": Station("Port Loop", -6.0000, -0.0001),
    "Port": Station("Port", 0.0000, 7.5000),
    "Thomas Yard": Station("Thomas Yard", 11.0000, 24.0000),
    "East Turner": Station("East Turner", 24.0000, 30.5000),
    "Barker": Station("Barker", 43.0000, 43.7500),
    "Canning": Station("Canning", 54.0000, 57.5000),
    "Chapman": Station("Chapman", 65.5000, 72.0000),
    "Durack": Station("Durack", 93.0000, 97.0000),
    "Kanga": Station("Kanga", 107.4000, 108.3000),
    "Forrest": Station("Forrest", 122.2500, 133.6000),
    "Hillside": Station("Hillside", 146.0000, 148.5940),
    "Gibb": Station("Gibb", 150.0840, 160.0000),
    "Coonarie": Station("Coonarie", 163.7500, 167.0000),
    "Nunna": Station("Nunna", 171.5000, 175.2500),
    "Hunter": Station("Hunter", 183.0000, 188.7000),
    "Avon": Station("Avon", 188.5000, 198.0000),
    "Summit": Station("Summit", 192.7500, 197.2500),
    "Morgan": Station("Morgan", 207.9000, 217.0000),
    "Bea Bea": Station("Bea Bea", 208.4100, 209.2770),
    "Bow": Station("Bow", 229.5000, 239.5000),
    "Maddina": Station("Maddina", 235.2000, 239.3500),
    "Cloudbreak": Station("Cloudbreak", 246.5000, 263.0000),
    "Capel": Station("Capel", 255.5000, 266.5000),
    "Jones": Station("Jones", 272.5700, 276.7000),
    "Christmas Creek": Station("Christmas Creek", 284.0000, 284.0000),
    "Firetail": Station("Firetail", 286.9050, 291.8590),
    "De Grey": Station("De Grey", 306.8410, 314.0000),
    "Duck": Station("Duck", 351.1290, 360.8260),
    "Eliwana": Station("Eliwana", 421.0000, 421.2000),
}


SECTIONS = {
    "Thomas": [
        Section("MLE_3.963_4.301", "ML Section 1", "Mainline1", 3.963, 4.301, "TL", "thomas", "main"),
        Section("MLE_4.301_5.971", "ML Section 1", "Mainline1", 4.301, 5.971, "TL", "thomas", "main"),
        Section("MLE_5.971_6.116", "ML Section 1", "Mainline1", 5.971, 6.116, "TL", "thomas", "main"),
        Section("MLE_6.116_7.343", "ML Section 1", "Mainline1", 6.116, 7.343, "TL", "thomas", "main"),
        
        # ** this appears to be be misnamed
        Section("MLB_30.022_43.538", "ML Section 1", "Mainline1", 7.343, 11.278, "TL", "thomas", "main"),

        Section("MLE_11.278_12.082", "ML Section 1", "Mainline1", 11.278, 12.082, "TL", "thomas", "main"),
        Section("MLE_12.082_15.255", "ML Section 1", "Mainline1", 12.082, 15.255, "TL", "thomas", "main"),
        Section("MLE_15.255_19.1", "ML Section 1", "Mainline1", 15.255, 19.1, "TL", "thomas", "main"),
        Section("MLE_19.1_19.728", "ML Section 1", "Mainline1", 19.1, 19.728, "TL", "thomas", "main"),
        Section("MLE_19.728_21.041", "ML Section 1", "Mainline1", 19.728, 21.041, "TL", "thomas", "main"),
        Section("MLE_21.041_22.907", "ML Section 1", "Mainline1", 21.041, 22.907, "TL", "thomas", "main"),
        Section("MLE_22.907_23.782", "ML Section 1", "Mainline1", 22.907, 23.782, "TL", "thomas", "main"),
        Section("MLE_23.782_27", "ML Section 1", "Mainline1", 23.782, 27, "TL", "thomas", "main"),
        
        Section("MLW_4.148_4.446", "ML Section 1", "Mainline1", 4.148, 4.446, "TL", "thomas", "bypass"),
        Section("MLW_4.446_5.825", "ML Section 1", "Mainline1", 4.446, 5.825, "TL", "thomas", "bypass"),
        Section("MLW_5.825_11.484", "ML Section 1", "Mainline1", 5.825, 11.484, "TL", "thomas", "bypass"),
        Section("MLW_11.484_11.876", "ML Section 1", "Mainline1", 11.484, 11.876, "TL", "thomas", "bypass"),
        Section("MLW_11.876_23.112", "ML Section 1", "Mainline1", 11.876, 23.112, "TL", "thomas", "bypass"),
        Section("MLW_23.112_23.577", "ML Section 1", "Mainline1", 23.112, 23.577, "TL", "thomas", "bypass"),
        Section("MLW_23.577_27", "ML Section 1", "Mainline1", 23.577, 27, "TL", "thomas", "bypass"),
    ],
    "Barker": [
        Section("MLE_27_30.022", "ML Section 2", "Mainline1", 27, 30.022, "ML", "mainline", "main"),
        Section("MLB_30.022_43.538", "ML Section 2", "Mainline1", 30.022, 43.538, "ML", "mainline", "main"),
        Section("MLB_43.538_54.227", "ML Section 2", "Mainline1", 43.538, 54.227, "ML", "mainline", "main"),
        Section("MLE_54.227_60", "ML Section 2", "Mainline1", 54.227, 60, "ML", "mainline", "main"),
        
        Section("MLW_27_30.022", "ML Section 2", "Mainline1", 27, 30.022, "ML", "mainline", "bypass"),
        Section("MLW_54.227_60", "ML Section 2", "Mainline1", 54.227, 60, "ML", "mainline", "bypass"),
    ],
    "Chapman": [
        Section("MLE_60_68.679", "ML Section 3", "Mainline1", 60, 68.679, "ML", "mainline", "main"),
        Section("MLE_68.679_71.266", "ML Section 3", "Mainline1", 68.679, 71.266, "ML", "mainline", "main"),
        Section("MLE_71.266_71.732", "ML Section 3", "Mainline1", 71.266, 71.733, "ML", "mainline", "main"),
        Section("MLB_71.734_86", "ML Section 3", "Mainline1", 71.733, 86, "ML", "mainline", "main"),
        
        Section("MLW_60_68.538", "ML Section 3", "Mainline1", 60, 68.538, "ML", "mainline", "bypass"),
        Section("MLW_68.538_71.732", "ML Section 3", "Mainline1", 68.538, 71.733, "ML", "mainline", "bypass"),
    ],
    "Forrest-I": [
        Section("MLB_86_93.677", "ML Section 4", "Mainline1", 86, 93.677, "ML", "mainline", "main"),
        Section("MLW_93.677_98.273", "ML Section 4", "Mainline1", 93.677, 98.273, "ML", "mainline", "main"),
        Section("MLW_98.273_107.575", "ML Section 4", "Mainline1", 98.273, 107.575, "ML", "mainline", "main"),
        Section("MLW_107.575_108.021", "ML Section 4", "Mainline1", 107.575, 108.021, "ML", "mainline", "main"),
        Section("MLW_108.021_121", "ML Section 4", "Mainline1", 108.021, 121, "ML", "mainline", "main"),
        
        Section("MLE_93.677_98.43", "ML Section 4", "Mainline1", 93.677, 98.429, "ML", "mainline", "bypass"),
        Section("MLE_98.429_107.782", "ML Section 4", "Mainline1", 98.429, 107.782, "ML", "mainline", "bypass"),
        Section("MLE_107.782_107.815", "ML Section 4", "Mainline1", 107.782, 107.815, "ML", "mainline", "bypass"),
        Section("MLE_107.815_121", "ML Section 4", "Mainline1", 107.815, 121, "ML", "mainline", "bypass"),
    ],
    "Forrest-II": [
        Section("MLW_121_125.765", "ML Section 5", "Mainline1", 121, 125.765, "ML", "mainline", "main"),
        Section("MLW_125.767_128.942", "ML Section 5", "Mainline1", 125.765, 128.943, "ML", "mainline", "main"),
        Section("MLW_128.943_133.315", "ML Section 5", "Mainline1", 128.943, 133.315, "ML", "mainline", "main"),
        Section("MLB_133.316_143", "ML Section 5", "Mainline1", 133.315, 143, "ML", "mainline", "main"),
        
        Section("MLE_121_125.906", "ML Section 5", "Mainline1", 121, 125.906, "ML", "mainline", "bypass"),
        Section("MLE_125.907_128.079", "ML Section 5", "Mainline1", 125.906, 128.079, "ML", "mainline", "bypass"),
        Section("MLE_128.079_128.828", "ML Section 5", "Mainline1", 128.079, 128.827, "ML", "mainline", "bypass"),
        Section("MLE_128.828_133.315", "ML Section 5", "Mainline1", 128.827, 133.315, "ML", "mainline", "bypass"),
    ],
    "Gibb": [
        Section("MLB_143_147.777", "ML Section 6", "Mainline1", 143, 147.777, "ML", "mainline", "main"),
        Section("MLB_147.777_153.084", "ML Section 6", "Mainline1", 147.777, 153.084, "ML", "mainline", "main"),
        Section("MLW_153.084_156.701", "ML Section 6", "Mainline1", 153.084, 156.701, "ML", "mainline", "main"),
        Section("MLB_156.701_163.917", "ML Section 6", "Mainline1", 156.701, 163.917, "ML", "mainline", "main"),
        Section("MLW_163.917_171", "ML Section 6", "Mainline1", 163.917, 171, "ML", "mainline", "main"),
        
        Section("PTE_153.085_156.701", "ML Section 6", "Mainline1", 153.084, 156.701, "ML", "mainline", "bypass"),
        
        Section("MLE_163.917_164.295", "ML Section 6", "Mainline1", 163.917, 164.295, "ML", "mainline", "bypass"),
        Section("MLE_164.295_171", "ML Section 6", "Mainline1", 164.295, 171, "ML", "mainline", "bypass"),
    ],
    "Nunna": [
        # note: no connections to Solomon line in here
        Section("MLW_171_174.319", "ML / SL Section 7", "Mainline2", 171, 174.319, "ML", "mainline", "main"),
        Section("MLW_174.319_174.577", "ML / SL Section 7", "Mainline2", 174.319, 174.577, "ML", "mainline", "main"),
        Section("MLB_174.577_174.652", "ML / SL Section 7", "Mainline2", 174.577, 174.652, "ML", "mainline", "main"),
        Section("MLB_174.652_183.267", "ML / SL Section 7", "Mainline2", 174.652, 183.267, "ML", "mainline", "main"),
        Section("MLW_183.267_186.462", "ML / SL Section 7", "Mainline2", 183.267, 186.462, "ML", "mainline", "main"),
        Section("MLB_186.462_192.978", "ML / SL Section 7", "Mainline2", 186.462, 192.978, "ML", "mainline", "main"),
        Section("MLB_192.978_197.033", "ML / SL Section 7", "Mainline2", 192.978, 197.033, "ML", "mainline", "main"),
        Section("MLB_197.033_203.001", "ML / SL Section 7", "Mainline2", 197.033, 203, "ML", "mainline", "main"),
        
        Section("MLE_171_174.577", "ML / SL Section 7", "Mainline2", 171, 174.577, "ML", "mainline", "bypass"),
        
        Section("PTE_183.267_184.723", "ML / SL Section 7", "Mainline2", 183.267, 184.723, "ML", "mainline", "bypass"),
        Section("PTE_184.723_185.421", "ML / SL Section 7", "Mainline2", 184.723, 185.422, "ML", "mainline", "bypass"),
        Section("PTE_185.421_186.462", "ML / SL Section 7", "Mainline2", 185.422, 186.462, "ML", "mainline", "bypass"),
        
        Section("PTE_192.978_197.033", "ML / SL Section 7", "Mainline2", 192.978, 197.033, "ML", "mainline", "bypass"),
    ],
    "Morgan": [
        Section("MLB_203_208.147", "ML Section 8", "Mainline2", 203, 208.147, "ML", "mainline", "main"),
        Section("MLW_208.147_214.913", "ML Section 8", "Mainline2", 208.147, 214.913, "ML", "mainline", "main"),
        Section("MLB_214.913_216.401", "ML Section 8", "Mainline2", 214.913, 216.401, "ML", "mainline", "main"),
        Section("MLB_216.401_235.452", "ML Section 8", "Mainline2", 216.401, 235.452, "ML", "mainline", "main"),
        Section("MLW_235.452_239.097", "ML Section 8", "Mainline2", 235.452, 239.097, "ML", "mainline", "main"),
        Section("MLB_239.097_245", "ML Section 8", "Mainline2", 239.097, 245, "ML", "mainline", "main"),
        
        Section("PTE_208.146_214.914", "ML Section 8", "Mainline2", 208.147, 214.913, "ML", "mainline", "bypass"),
        Section("PTE_235.452_235.839", "ML Section 8", "Mainline2", 235.452, 235.839, "ML", "mainline", "bypass"),
        Section("PTE_235.839_239.097", "ML Section 8", "Mainline2", 235.839, 239.097, "ML", "mainline", "bypass"),
    ],
    "Cloudbreak": [
        Section("MLB_245_249.608", "ML Section 9", "Mainline2", 245, 249.608, "ML", "mainline", "main"),
        Section("MLB_249.608_253.147", "ML Section 9", "Mainline2", 249.608, 253.147, "ML", "mainline", "main"),
        Section("MLB_253.147_255.676", "ML Section 9", "Mainline2", 253.147, 255.676, "ML", "mainline", "main"),
        Section("MLB_255.676_256.5", "ML Section 9", "Mainline2", 255.676, 256.5, "ML", "mainline", "main"),
        Section("MLW_256.5_259.854", "ML Section 9", "Mainline2", 256.5, 259.854, "ML", "mainline", "main"),
        Section("MLB_259.854_261.705", "ML Section 9", "Mainline2", 259.854, 261.705, "ML", "mainline", "main"),
        Section("MLB_261.705_270.001", "ML Section 9", "Mainline2", 261.705, 270, "ML", "mainline", "main"),
        
        Section("CBM_249.609_252.134", "ML Section 9", "Mainline2", 249.608, 252.135, "ML", "mainline", "bypass"),
        Section("CBM_252.134_253.295", "ML Section 9", "Mainline2", 252.135, 253.295, "ML", "mainline", "bypass"),
        Section("CBM_253.295_254.006", "ML Section 9", "Mainline2", 253.295, 254.000, "ML", "mainline", "bypass"),
        Section("CBM_254.006_254.733", "ML Section 9", "Mainline2", 254.000, 254.732, "ML", "mainline", "bypass"),
        Section("CBM_254.733_254.824", "ML Section 9", "Mainline2", 254.732, 254.830, "ML", "mainline", "bypass"),
        Section("CBM_254.824_256.304", "ML Section 9", "Mainline2", 254.830, 256.283, "ML", "mainline", "bypass"),

        Section("PTE_256.5_259.855", "ML Section 9", "Mainline2", 256.5, 259.854, "ML", "mainline", "bypass"), 
    ],
    "Jones": [
        Section("MLB_270_272.823", "ML Section 10", "Mainline2", 270, 272.823, "ML", "mainline", "main"),
        Section("MLE_272.823_276.409", "ML Section 10", "Mainline2", 272.823, 276.409, "ML", "mainline", "main"),
        Section("MLB_276.409_286.707", "ML Section 10", "Mainline2", 276.409, 286.707, "ML", "mainline", "main"),
        Section("MLB_286.707_287.264", "ML Section 10", "Mainline2", 286.707, 287.264, "ML", "mainline", "main"),
        Section("CCL_287.264_287.521", "ML Section 10", "Mainline2", 287.264, 287.520, "ML", "mainline", "main"),
        Section("CCL_287.521_294.219", "ML Section 10", "Mainline2", 287.520, 294.218, "ML", "mainline", "main"),
        Section("CCL_294.219_297.315", "ML Section 10", "Mainline2", 294.218, 297.314, "ML", "mainline", "main"),
        
        Section("PTW_272.823_273.077", "ML Section 10", "Mainline2", 272.823, 273.077, "ML", "mainline", "bypass"),
        Section("PTW_273.077_276.409", "ML Section 10", "Mainline2", 273.077, 276.409, "ML", "mainline", "bypass"),
        
        Section("PTW_286.707_287.521", "ML Section 10", "Mainline2", 286.707, 287.520, "ML", "mainline", "bypass"),
        # end of mainline (Christmas creek)
    ],
    "Nunna-2": [
        # start of Solomon line
        Section("SLB_174.319_174.857", "ML Section L7", "Soloman", 174.319, 174.857, "SL", "solomon", "main"),
        Section("SLB_174.857_175", "ML Section L7", "Soloman", 174.857, 175, "SL", "solomon", "main"),
        Section("SLB_175_181", "ML Section L7", "Soloman", 175, 181, "SL", "solomon", "main"),
    ],
    "Avon": [
        Section("SLB_181_191.459", "Section SL7", "Soloman", 181, 191.459, "SL", "solomon", "main"),
        Section("SLE_191.459_195.023", "Section SL7", "Soloman", 191.459, 195.023, "SL", "solomon", "main"),
        Section("SLB_195.023_208.428", "Section SL7", "Soloman", 195.023, 208.428, "SL", "solomon", "main"),
        Section("SLB_208.428_209.260", "Section SL7", "Soloman", 208.428, 209.259, "SL", "solomon", "main"),
        Section("SLB_209.260_214", "Section SL7", "Soloman", 209.259, 214, "SL", "solomon", "main"),
        
        Section("SPW_191.459_191.711", "Section SL7", "Soloman", 191.459, 191.712, "SL", "solomon", "bypass"),
        Section("SPW_191.711_195.023", "Section SL7", "Soloman", 191.712, 195.023, "SL", "solomon", "bypass"),
    ],
    "Bow": [
        Section("SLB_214_232.687", "Section SL8", "Soloman", 214, 232.687, "SL", "solomon", "main"),
        Section("SLW_232.687_236.296", "Section SL8", "Soloman", 232.687, 236.296, "SL", "solomon", "main"),
        Section("SLB_236.296_248", "Section SL8", "Soloman", 236.296, 248, "SL", "solomon", "main"),
        
        Section("SPE_232.687_232.94", "Section SL8", "Soloman", 232.687, 232.94, "SL", "solomon", "bypass"),
        Section("SPE_232.94_236.297", "Section SL8", "Soloman", 232.94, 236.297, "SL", "solomon", "bypass"),
    ],
    "Capel": [
        Section("SLB_248_259.416", "Section SL9", "Soloman", 248, 259.416, "SL", "solomon", "main"),
        Section("SLE_259.416_262.853", "Section SL9", "Soloman", 259.416, 262.853, "SL", "solomon", "main"),
        Section("SLB_262.854_263.296", "Section SL9", "Soloman", 262.853, 263.297, "SL", "solomon", "main"),
        Section("SLB_263.296_281", "Section SL9", "Soloman", 263.297, 281, "SL", "solomon", "main"),
        
        Section("SPW_259.416_262.854", "Section SL9", "Soloman", 259.416, 262.853, "SL", "solomon", "bypass"),
    ],
    "Firetail": [
        Section("SLB_281_288.709", "SL / EL Section SL10", "Soloman", 281, 288.709, "SL", "solomon", "main"),
        
        Section("ELB_288.709_309.841", "SL / EL Section SL10", "Soloman", 288.709, 291.925, "SL", "solomon", "loop"),
        Section("SML_291.925_301.408", "SL / EL Section SL10", "Soloman", 291.925, 301.408, "SL", "solomon", "loop"),
        Section("SML_301.408_301.601", "SL / EL Section SL10", "Soloman", 301.408, 301.601, "SL", "solomon", "loop"),
        Section("SML_301.601_302.492", "SL / EL Section SL10", "Soloman", 301.601, 302.492, "SL", "solomon", "loop"),
        Section("SML_302.492_303.272", "SL / EL Section SL10", "Soloman", 302.492, 303.272, "SL", "solomon", "loop"),
    ],
    "De Grey": [
        Section("ELW_288.709_309.844", "Section EL11", "Eliwana", 288.709, 309.844, "EL", "eliwana", "main"),
        Section("ELB_309.844_313.313", "Section EL11", "Eliwana", 309.844, 313.313, "EL", "eliwana", "main"),
        Section("ELB_313.313_332.927", "Section EL11", "Eliwana", 313.313, 332.926, "EL", "eliwana", "main"),
        
        Section("ELW_309.844_313.313", "Section EL11", "Eliwana", 309.844, 313.313, "EL", "eliwana", "bypass"),
    ],
    "Duck": [
        Section("ELW_332.926_354.13", "Section EL12", "Eliwana", 332.926, 354.129, "EL", "eliwana", "main"),
        Section("ELB_354.129009_357.826", "Section EL12", "Eliwana", 354.129, 357.826, "EL", "eliwana", "main"),
        Section("ELB_357.825948_373.117", "Section EL12", "Eliwana", 357.826, 373.117, "EL", "eliwana", "main"),
    ],
    "Future": [
        Section("ELB_373.11699_423.483", "Section EL13", "Eliwana", 373.117, 423.482, "EL", "eliwana", "main"),
        Section("SLB_423.482018_423.978022", "Section EL13", "Eliwana", 423.482, 423.978, "EL", "eliwana", "main"),
    ],
    "Eliwana": [
        Section("EL1_423.978_431.634", "Section EL14", "Eliwana", 423.978, 431.533, "EL", "eliwana", "main"),
    ]
}


def get_station(chainage: float) -> str:
    """Get the station containing the given chainage."""
    for station in STATIONS.values():
        if station.start_km is not None and station.end_km is not None:
            if station.start_km <= chainage <= station.end_km:
                return station.name
    return ""


def get_section(chainage: float, line_code: str, line_class: str = "") -> list:
    """Get the section containing the given chainage."""
    valid_sections = []
    for name, components in SECTIONS.items():
        for component in components:
            if component.line_code != line_code:
                continue
            if component.start_km <= chainage <= component.end_km:
                if line_class and line_class != component.line_class:
                    continue
                valid_sections.append((name, component.section_id, component.asset_name))
    return valid_sections


def get_section_attribute(asset_name: str, attr: str = None):
    """
    For a given asset_name determine whether it corresponds to a main line, bypass or other
    """
    for _, components in SECTIONS.items():
        for component in components:
            if component.asset_name == asset_name:
                if attr is None:
                    return component
                
                if hasattr(component, attr):
                    return getattr(component, attr)
                else:
                    logging.error(f"Section class has no attribute: {attr}")
                    return None
    return None


def get_section_metadata(asset_name: str):
    section = get_section_attribute(asset_name)
    if section:
        return pd.Series({
            "line": section.line,
            "line_code": section.line_code,
            "line_class": section.line_class
        })
    else:
        prefix = asset_name.split("_", 1)[0]
        if prefix in VALID_LINE_PREFIX:
            logging.debug(f"No section found for asset_name: {asset_name}")
        return pd.Series({"line": None, "line_code": None, "line_class": None})


VALID_LINE_PREFIX = {
    "MLB", "MLW", "MLE", "ELB", "ELW", "EL1", "SLB", "SLW", "SLE", "SML", "SPW", "SPE", "PTW", "CCL", "CBM", "PTE",
}