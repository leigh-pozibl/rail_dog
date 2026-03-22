"""
Main entry point for rail dog
"""
import click
import os
import logging
import sys
from dotenv import load_dotenv

from snappy_utils.base import setup_workflow
from snappy_utils.params import DBConnection

from rail_dog.processor import Processor
from rail_dog.report import Report
from rail_dog.utils.io_utils import load_config_file, load_json_blob
# from rail_dog.utils.report_utils import generate_excel_report

load_dotenv()

DUMMY_ID = "00000000-0000-0000-0000-000000000000"

@click.command()
@click.option('--config', default=None, type=str, help="The main config file")
@click.option('--db-env', default=None, type=str,
              help="If using database choose from 'local' or 'prod' or else provide the full db connection string")
@click.option('--project-id', default=DUMMY_ID, type=str, help="The project id")
@click.option('--output-dir', default="output", type=str, help="The root output directory")
@click.option('--json-input', default=None, type=str, help="Alternative input via json blob")
def main(config, db_env, project_id, output_dir, json_input):

    if config is None and json_input is None:
        print("No input configs provided, exiting.")
        return

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    workflow_name = "rail_dog"
    metadata = setup_workflow(workflow_name, project_id, DUMMY_ID, DUMMY_ID, output_dir, db_env)

    if config:
        root_path = os.path.dirname(os.path.abspath(config))
        params, db = load_config_file(config, root_path, metadata, db_env)
    else:
        params, db = load_json_blob(json_input, metadata, db_env)

    params.output_dir = output_dir
    
    # setup connection to postgres database
    if db_env in {"local"}:
        db_env = os.environ.get("LOCAL_CONNECTION_STRING")
        if db_env:
            db = DBConnection(db_env)

    logging.info("Finished loading input configuration files")

    process_dir = os.path.join(params.output_dir, "process")
    if not os.path.exists(process_dir):
        os.mkdir(process_dir)

    logging.info("Starting processing")
    pr = Processor(params, db, metadata, output_dir=process_dir)
    pr.run()
    pr.write_outputs()
    
    if params.execution.process_only:
        logging.info("Process only flag set, skipping report generation")
        return
    
    logging.info("Generating xlsx report")
    report_dir = os.path.join(params.output_dir, "reports")
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)
    
    mapbox_token = os.environ.get("MAPBOX_TOKEN", "")

    if params.execution.global_analysis_date:
        logging.info(f"Using global date from execution config: {params.execution.global_analysis_date}")
        model = Report(db.engine, report_dir, global_date=params.execution.global_analysis_date, mapbox_token=mapbox_token)
    else:
        model = Report(db.engine, report_dir, analysis_dates=params.base_data.analysis_dates, mapbox_token=mapbox_token)
    model.write_excel_report()


if __name__ == "__main__":
    sys.exit(main())
