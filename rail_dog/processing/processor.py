import os
import logging

import pyproj

from shapely.strtree import STRtree

from snappy_utils.params import Metadata, DBConnection
from snappy_utils.geom_utils import utm_crs_from_a_geom

from rail_dog.utils.db_utils import get_table_data
from rail_dog.configs.params import BaseConfiguration
from rail_dog.configs.thresholds import RAIL_DEGRADATION_THRESHOLDS
from rail_dog.utils.trend_utils import compute_trend

from rail_dog.processing.geometry import GeometryMixin
from rail_dog.processing.curves import CurvesMixin
from rail_dog.processing.assets import AssetsMixin
from rail_dog.processing.track_data import TrackDataMixin
from rail_dog.processing.ensco import EnscoMixin
from rail_dog.processing.condition import ConditionMixin


class Processor(GeometryMixin, CurvesMixin, AssetsMixin, TrackDataMixin, EnscoMixin, ConditionMixin):
    def __init__(self, params: BaseConfiguration, db: DBConnection, metadata: Metadata, output_dir: str):

        self.db = db
        self.data = params.base_data
        self.controls = params.controls
        self.params = params.parameters.preprocess
        self.input_fields = params.parameters.input_fields
        self.costs = params.parameters.costs
        self.get_filters = params.parameters.preprocess.get_filters
        self.metadata = metadata

        self.segment_method = self.params.segment_method
        self.segment_length = 100

        self.working_crs = self.get_working_crs(params.parameters.globals.working_crs)
        params.parameters.globals.working_crs = self.working_crs
        logging.info(f"Working CRS is: {self.working_crs}")

        # convert all data to a common working_crs
        self.use_centroid = None
        for layer in self.data.active_layers:
            layer_data = self.data.get_data(layer)
            layer_data.geometry = layer_data.geometry.force_2d()
            self.data.set_data(layer, layer_data.to_crs(self.working_crs))

        # some mappings
        self.index_to_node_id = dict()
        self.index_to_edge_id = dict()
        self.cand_id_to_solution_id = dict()
        self.trees = {}

        self.debug = True

        self.refresh_data = params.base_data.refresh_data

        src = pyproj.CRS(self.working_crs)
        dst = pyproj.CRS('epsg:4326')
        self.PROJ = pyproj.Transformer.from_crs(src, dst, always_xy=True)

        self.h3_resolution = 15

        self.output_dir = output_dir
        self.output_fmt = params.parameters.globals.output_fmt

        self.line_name_to_code = {
            "Mainline": "ML",
            "Thomas": "TL",
            "Eliwana": "EL",
            "Solomon": "SL",
        }
        self.line_code_to_name = {v: k for k, v in self.line_name_to_code.items()}

        # self.THOMAS_START = -3.7
        self.THOMAS_START = -3.7515
        self.THOMAS_END = 26.9
        self.SOLOMON_START = 174
        self.ELIWANA_START = 288.70
        self.START_OFFSET = {
            "TL": self.THOMAS_START,
            "ML": self.THOMAS_END,
            "SL": self.SOLOMON_START,
            "EL": self.ELIWANA_START,
        }

        self.THRESHOLDS = RAIL_DEGRADATION_THRESHOLDS()

    def get_working_crs(self, input_crs) -> str:
        """
        It is better to use a UTM crs that is localised to the area being processed.
        nb: using the epsg:3857 crs will report erroneous lengths for linestrings
        """
        if input_crs.split(":")[0] == "epsg":
            return str(input_crs)

        elif input_crs == "utm":
            for layer in self.data.active_layers:
                try:
                    layer_data = self.data.get_data(layer)
                    sample_geom = layer_data["geometry"][0]
                    crs = utm_crs_from_a_geom(sample_geom)
                    return f"epsg:{crs.to_epsg()}"
                except:
                    pass
            logging.error("Couldn't determine a localised projection, using default epsg:3857")
            return "epsg:3857"

        else:
            logging.error("Couldn't determine a localised projection, using default epsg:3857")
            return "epsg:3857"

    def run(self):

        # process the rail geometry input and split into separate lines
        self.process_base_geometry()

        ############################
        logging.info("############################")
        if self.refresh_data("BASE_SEGMENTS"):
            logging.info(f"Segmenting into {self.segment_length}m chainage intervals")
            self.process_into_chainage_segments(self.segment_length)
        else:
            logging.info("Using existing BASE_SEGMENTS data from database")
            self.segment_data = {}
            for line_code in self.line_code_to_name.keys():
                segment_gdf = get_table_data(
                    self.db.engine,
                    crs=self.working_crs,
                    query=f"SELECT * FROM rail_segments WHERE line_code = '{line_code}'"
                )
                _geoms = list(segment_gdf.geometry)
                self.segment_data[line_code] = (STRtree(_geoms), segment_gdf)

        ############################
        logging.info("############################")
        logging.info("Processing SAP records")
        if self.refresh_data("SAP"):
            self.process_sap_records()
        else:
            logging.info("Using existing SAP data from database")
            # we only need sap data when refreshing assets

            # sap_date = self.data.analysis_dates.get("SAP")
            # if not sap_date:
            #     logging.error("An analysis date for SAP data not specified")
            #     exit()
            # self.sap_records = get_table_data(
            #     self.db.engine,
            #     query=f"SELECT * FROM sap_records WHERE collection_date = '{sap_date}"
            # )

        ############################
        logging.info("############################")
        logging.info("Processing assets")
        if self.refresh_data("ASSETS"):
            self.create_assets()
        else:
            logging.info("Using existing asset data from database")
        self.assets = get_table_data(self.db.engine, table_name="assets", crs=self.working_crs)

        ############################
        logging.info("############################")
        logging.info("Processing curves & tangent sections")
        if self.refresh_data("CURVE_SECTIONS"):
            self.create_curve_sections()
        else:
            logging.info("Using existing curves data from database")
        self.curve_sections = get_table_data(self.db.engine, table_name="rail_sections", crs=self.working_crs)

        ############################
        logging.info("############################")
        logging.info("Aggregating assets to chainage segments")
        if self.refresh_data("AGG_ASSETS"):
            self.aggregate_assets_to_segments()
        else:
            logging.info("Using existing aggregated asset data from database")

        ############################
        logging.info("############################")
        logging.info("Processing GBFI data")
        if self.refresh_data("GBFI"):
            collection_date = self.data.collection_dates.get("GBFI_ML")
            db_action = self.data.db_actions.get("GBFI_ML")
            if collection_date is None:
                logging.error("Must specify a collection date for the GBFI dataset being loaded.")
                return
            self.process_gbfi_data(collection_date, db_action=db_action)
            self.aggregate_gbfi_to_segments(collection_date, db_action=db_action)
        else:
            logging.info("Using existing GBFI data from database")
            gbfi_date = self.data.analysis_dates.get("GBFI")
            if not gbfi_date:
                logging.error("An analysis date for GBFI data not specified")
                exit()
            self.agg_gbfi = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_gbfi WHERE collection_date = '{gbfi_date}'"
            )

        ############################
        logging.info("############################")
        logging.info("Processing Moisture data")
        if self.refresh_data("MOI"):
            collection_date = self.data.collection_dates.get("MOI_ML")
            db_action = self.data.db_actions.get("MOI_ML")
            if collection_date is None:
                logging.error("Must specify a collection date for the Moisture dataset being loaded.")
                return
            self.process_moisture_data(collection_date, db_action=db_action)
            self.aggregate_moisture_to_segments(collection_date, db_action=db_action)
        else:
            logging.info("Using existing Moisture data from database")
            moisture_date = self.data.analysis_dates.get("MOI")
            if not moisture_date:
                logging.error("An analysis date for Moisture data not specified")
                exit()
            self.agg_moisture = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_moisture WHERE collection_date = '{moisture_date}'"
            )

        ############################
        logging.info("############################")
        logging.info("Processing Ballast data")
        if self.refresh_data("BALL"):
            collection_date = self.data.collection_dates.get("BALL_ML")
            db_action = self.data.db_actions.get("BALL_ML")
            if collection_date is None:
                logging.error("Must specify a collection date for the Ballast dataset being loaded.")
                return
            self.process_ballast_data(collection_date, db_action=db_action)
            self.aggregate_ballast_to_segments(collection_date, db_action=db_action)
        else:
            logging.info("Using existing Ballast data from database")
            ballast_date = self.data.analysis_dates.get("BALL")
            if not ballast_date:
                logging.error("An analysis date for Ballast data not specified")
                exit()
            self.agg_ballast = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_ballast WHERE collection_date = '{ballast_date}'"
            )

        ############################
        logging.info("############################")
        logging.info("Processing TSR data")
        if self.refresh_data("TSR"):
            collection_date = self.data.collection_dates.get("TSR_ML")
            db_action = self.data.db_actions.get("TSR_ALL")
            self.process_tsr_data(collection_date=collection_date, db_action=db_action)
        else:
            logging.info("Using existing TSR data from database")

        ############################
        logging.info("############################")
        logging.info("Processing TQI data")
        if self.refresh_data("TQI"):
            collection_date = self.data.collection_dates.get("TQI")
            if collection_date is not None:
                db_action = self.data.db_actions.get("TQI")
                self.process_tqi_data(collection_date, db_action)
                self.aggregate_tqi_to_segments(collection_date, db_action, line_class='main')
                compute_trend(self.db.engine, "tqi", collection_date)
            elif self.data.analysis_dates.get("TQI") is not None:
                logging.info("Refresh set with no collection date provided for TQI, but analysis date exists.")
                logging.info("Updating TQI trends using existing TQI data in database")
                compute_trend(self.db.engine, "tqi", self.data.analysis_dates["TQI"])
            else:
                logging.error("Must specify a collection date for the TQI dataset being loaded.")
                return
        else:
            logging.info("Using existing TQI data from database")
            tqi_date = self.data.analysis_dates.get("TQI")
            if not tqi_date:
                logging.error("An analysis date for TQI data not specified")
                return
            self.agg_tqi = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_tqi WHERE collection_date = '{tqi_date}'"
            )

        ############################
        logging.info("############################")
        logging.info("Processing DTR data")
        if self.refresh_data("DTR"):
            collection_date = self.data.collection_dates.get("DTR")
            db_action = self.data.db_actions.get("DTR")
            if collection_date is None:
                logging.error("Must specify a collection date for the DTR dataset being loaded.")
                return
            self.process_dtr_data(collection_date, db_action=db_action)
            self.aggregate_dtr_to_segments(collection_date, line_class='main', db_action=db_action)
        else:
            logging.info("Using existing DTR data from database")
            dtr_date = self.data.analysis_dates.get("DTR")
            if not dtr_date:
                logging.error("An analysis date for DTR data not specified")
                return
            self.agg_dtr = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_dtr WHERE collection_date = '{dtr_date}'"
            )

        ############################
        logging.info("############################")
        if self.data.rp_data is not None:
            logging.info("Processing RP data")
            self.process_rp_data_into_sections()
            self.calculate_rp_stats_per_section()

        ############################
        logging.info("############################")
        if self.refresh_data("TG"):
            if self.data.tg_data is not None:
                logging.info("Processing TG data")
                self.process_tg_data(db_action="append")
            else:
                logging.error("No input TG data provided")
        else:
            logging.info("Using existing TG data from database")

    def write_outputs(self):
        logging.info("Writing outputs")
        path_out = os.path.join(self.output_dir, f"thomas_segments.{self.output_fmt}")
        gdf = self.segment_data["TL"][1]
        gdf.to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"mainline_segments.{self.output_fmt}")
        gdf = self.segment_data["ML"][1]
        self.segment_data["ML"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"solomon_segments.{self.output_fmt}")
        gdf = self.segment_data["SL"][1]
        self.segment_data["SL"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"eliwana_segments.{self.output_fmt}")
        gdf = self.segment_data["EL"][1]
        self.segment_data["EL"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"pois.{self.output_fmt}")
        self.poi_gdf.to_crs('epsg:4326').to_file(path_out)
        path_out = os.path.join(self.output_dir, f"assets.{self.output_fmt}")
        self.assets.to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"curve_sections.{self.output_fmt}")
        self.curve_sections.to_crs('epsg:4326').to_file(path_out)

        # path_out = os.path.join(self.output_dir, f"rp_data.{self.output_fmt}")
        # self.rp_processed.to_crs('epsg:4326').to_file(path_out)
        # path_out = os.path.join(self.output_dir, f"rp_data.csv")
        # self.rp_processed.to_crs('epsg:4326').to_csv(path_out)
