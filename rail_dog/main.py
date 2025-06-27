"""
Main entry point for snappy cat
"""
import click
import os
import logging
import sys
from dotenv import load_dotenv

from rail_dog.processor import Processor
# from snappy_cat.solvers.solve import setup_solver
from rail_dog.output import OutputWriter
from rail_dog.utils.io_utils import load_config_file, load_json_blob
from snappy_utils.params import Metadata

load_dotenv()


@click.command()
@click.option('--config', default=None, type=str, help="The main config file")
@click.option('--db-env', default=None, type=str,
              help="If using database choose from 'local' or 'prod' or else provide the full db connection string")
@click.option('--project-id', default="00000000-0000-0000-0000-000000000000", type=str, help="The project id")
@click.option('--output-dir', default="output", type=str, help="The root output directory")
@click.option('--json-input', default=None, type=str, help="Alternative input via json blob")
def main(config, db_env, project_id, output_dir, json_input):

    if config is None and json_input is None:
        print("No input configs provided, exiting.")
        return

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'snappy.log'),  mode='w'),
            logging.StreamHandler()
        ]
    )

    if os.environ.get('PYTHONHASHSEED'):
        logging.info(f"PYTHONHASHSEED: {os.environ['PYTHONHASHSEED']}")

    # these credentials refer to the project named "Test Project 01 - acc:Pozibl-user:Admin"
    metadata = Metadata(
        project_id="62735f6f-3e76-4787-bc4d-5b3d6f5b6ed0",
        owner_id="419bc811-29b4-4768-9a2e-4e8212ea6ef5",
    )

    if config:
        root_path = os.path.dirname(os.path.abspath(config))
        params, db = load_config_file(config, root_path, metadata, db_env)
    else:
        params, db = load_json_blob(json_input, metadata, db_env)

    params.output_dir = output_dir

    logging.info("Finished loading input configuration files")

    if params.execution.base_graph is None and params.execution.solution_graph is None:
        logging.info("Starting processing")

        process_dir = os.path.join(params.output_dir, "process")
        if not os.path.exists(process_dir):
            os.mkdir(process_dir)

        pr = Processor(params, db, metadata, output_dir=process_dir)
        pr.run()
        # base_graph = pr.bg.graph
    #     solution_graph = pr.solution_graph
    # else:
    #     base_graph = params.execution.base_graph
    #     solution_graph = params.execution.solution_graph

    #     input_fields = params.parameters.input_fields
    #     if "addresses" in params.base_data.active_layers:
    #         input_fields["demand"] = input_fields["addresses"]
    #     elif "parcels" in params.base_data.active_layers:
    #         input_fields["demand"] = input_fields["parcels"]

    steps_to_run = [s for s in params.execution.run_steps if s != "pp"]
    logging.info(f"Running steps: {steps_to_run}")
    if len(steps_to_run) > 0:
        pruned_base_graph = None
        for step, solve_config in enumerate(params.parameters.solve_configs):
            if solve_config.name not in steps_to_run:
                continue

            base_solve_dir = os.path.join(params.output_dir, "solution")
            if not os.path.exists(base_solve_dir):
                os.mkdir(base_solve_dir)

            solve_dir = os.path.join(base_solve_dir, solve_config.name)
            if not os.path.exists(solve_dir):
                os.mkdir(solve_dir)

            raw_solution = params.execution.raw_solution.get(solve_config.name)

            logging.info("Create solver")
            sl = setup_solver(
                pruned_base_graph if pruned_base_graph else base_graph,
                solution_graph,
                params.parameters.architecture,
                solve_config,
                raw_solution,
                solve_dir,
                params.parameters.globals.output_fmt,
                db,
                metadata,
            )

            logging.info("Running solve")
            sl.solve()

            if sl.pruned_base_graph:
                pruned_base_graph = sl.pruned_base_graph
            else:
                pruned_base_graph = None

        solution_graph = sl.solution_graph
    else:
        # not running any solve
        pass

    # logging.info("Writing final outputs")
    # output_dir = os.path.join(params.output_dir, "output")
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    # op = OutputWriter(solution_graph, base_graph, params, output_dir=output_dir)
    # op._set_db(db, metadata)
    # op.write_gis_output()


if __name__ == "__main__":
    sys.exit(main())
